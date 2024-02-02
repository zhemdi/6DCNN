/*************************************************************************\

Sergei Grudinin, 2012
Mikhail Karasikov, 2016
All Rights Reserved.

\**************************************************************************/

#include "engine.hpp"

//#include "bessel.h" // for bessel functions
//#include "besselfrac.h" // for bessel functions
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <cstdio>

#include "cClassicalParserPDB.hpp"
#include "cGrid.hpp"
#include "cAlgorithmTemplates.hpp"
#include "cRotamerLibrary.hpp"
#include "mathFunctions.hpp"
#include "cSparseMatrixOutput.hpp"
#include "energyModel.hpp"
#include "cProteinFeaturizer.hpp"
#include "cProteinMapper.hpp"
#include "SolventGrid.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include "npy.hpp"
#include <time.h>

#include "cSH.hpp" //for SH and oriented graphs
#include "cMemoryManager.cpp" // it's a template

const size_t kMaxRotamersPerResidue = 5;
const size_t kNumIterations = 10;


size_t current_resInd = 0;

using std::vector;
using std::set;


bool if_target_res_exists(const std::pair<size_t, size_t> element){
    return element.second == current_resInd;
}

bool write_vector(const Eigen::SparseVector<double> &vector,
                  const std::string &filename) {
  cSparseMatrixOutputMAT matrixOutput(vector.size(), 1);
  if (!matrixOutput.initialize(filename))
    return false;

  for (Eigen::SparseVector<double>::InnerIterator it(vector); it; ++it) {
    if (!matrixOutput.writeTriplet(it.index(), 0, it.value()))
      return false;
  }
  return matrixOutput.deinitialize();
}


bool writeMap(std::fstream & binOutputFile, const uint8_t* map, const int32_t* meta, size_t mapSize, size_t metaSize) {
	binOutputFile.write((char*)map, mapSize);
	binOutputFile.write((char*)meta, metaSize);
	return true;
}

bool writeMap(FILE* binOutputFile, const uint8_t* map, const int32_t* meta, size_t mapSize, size_t metaSize) {
	fwrite(map, sizeof(char), mapSize, binOutputFile);
	fwrite(meta, sizeof(char), metaSize, binOutputFile);
	return true;
}

// bool writeAtomFeatures(cProtein *protein, cProtein *target_protein, const int order, const std::string &general, const std::string &expansions, const std::string &outputFirstSlaterMatrices, const std::string &outputBesselMatrices, const std::string &outputSecondSlaterMatrices, const std::string &neighbfunFileName, const std::string &edges, const std::string &directions, const std::string &scoreoutputFilename, const std::string &edgestypesFilename, const std::string &scoreFilename, const std::string &sphNodesFilename, const std::string &score_type, bool native, Real Radius, Real Radius2, Real maxQ, Real sigma, bool orth, bool add_solvent, int resgap, bool add_sph_harm_bn_nodes, bool use_aggregation_tensors, bool use_bessel_matrices, bool use_neighborhood_function_coefs) {
bool writeAtomFeatures(cProtein *protein, cProtein *target_protein, const int order, const std::string &general, const std::string &expansions, const std::string &outputFirstSlaterMatrices, const std::string &outputBesselMatrices, const std::string &outputSecondSlaterMatrices, const std::string &edges, const std::string &directions, const std::string &scoreoutputFilename, const std::string &edgestypesFilename, const std::string &scoreFilename, const std::string &sphNodesFilename, const std::string &score_type, bool native, Real Radius, Real Radius2, Real maxQ, Real sigma, bool orth, bool add_solvent, int resgap, bool add_sph_harm_bn_nodes, bool use_aggregation_tensors, bool use_bessel_matrices) {


    //FIXME a proper value can be chosen here...
    const int nFourierShells = order;
    // Real maxQ = 1.5; //// Need a proper definition of this....
    // Real maxQ = 2*M_PI*(nFourierShells - 1)/Radius;
    Real maxR = Radius;
    // Real Radius2 = 12.0;

    //allocate memory + precompute constants
    
     std::vector<std::unique_ptr<cWaterResidue>> solvent_residues; 
    int num_types;
    if (add_solvent){
        num_types = 168;
        solvent_residues = GetHydrationShell(protein, 1.5, 3.0, Radius);
    }else{
        num_types = 167;
    }
    

    

    cSH sh;
    sh.init(order, num_types);
    sh.initWigner(order, order);

    std::vector <size_t> edges_vector;

    std::vector <cVector3> directions_vector;
    std::vector <int> scores_vector;

    std::vector <int> edges_types;
    std::vector <int> res_types;

    std::map<std::string, int> res_map;

    res_map["ALA"] = 0;
	  res_map["ARG"] = 1;
	  res_map["ASN"] = 2;
	  res_map["ASP"] = 3;
	  res_map["CYS"] = 4;
	  res_map["GLN"] = 5;
	  res_map["GLU"] = 6;
	  res_map["GLY"] = 7;
	  res_map["HIS"] = 8;
	  res_map["ILE"] = 9;
	  res_map["LEU"] = 10;
	  res_map["LYS"] = 11;
	  res_map["MET"] = 12;
	  res_map["PHE"] = 13;
    res_map["PRO"] = 14;
	  res_map["SER"] = 15;
	  res_map["THR"] = 16;
	  res_map["TRP"] = 17;
	  res_map["TYR"] = 18;
	  res_map["VAL"] = 19;
	  res_map["SEC"] = 15;
	  res_map["MSE"] = 12;
	  res_map["AMINO_TYPE_END"] = 20;

    std::vector <size_t> neighbours;

    int num_at, num_sa;

    

    std::vector <cVector3> recomputed_directions_vector;

    sComplex *WignerPhaseA = new sComplex[order];
    sComplex *WignerPhaseG = new sComplex[order];
    Real        ***WignerD;
    cMemoryManager::allocWignerRotation(WignerD, order);
    Real    **SphBesPoly;
    cMemoryManager::alloc2DArray(SphBesPoly, nFourierShells, order);

    cAtomicResidueMapper A(167, 12, 1.0, 1.6);



    FILE * generalFile = fopen(general.c_str(),"wb");
    if (generalFile == NULL){
        std::cerr << "The general file is not open" << std::endl;
        exit(1);
    }
    FILE * expansionsFile = fopen(expansions.c_str(),"wb");
    if (expansionsFile == NULL){
        std::cerr << "The nodes file is not open" << std::endl;
        exit(1);
    }
    FILE * outputFirstSlaterMatricesFile = fopen(outputFirstSlaterMatrices.c_str(),"wb");
    if (outputFirstSlaterMatricesFile == NULL){
        std::cerr << "The fsm file is not open" << std::endl;
        exit(1);
    }
    FILE * outputBesselMatricesFile = fopen(outputBesselMatrices.c_str(),"wb");
    if (outputBesselMatricesFile == NULL){
        std::cerr << "The bm file is not open" << std::endl;
        exit(1);
    }
    FILE * outputSecondSlaterMatricesFile = fopen(outputSecondSlaterMatrices.c_str(),"wb");
    if (outputSecondSlaterMatricesFile == NULL){
        std::cerr << "The ssm file is not open" << std::endl;
        exit(1);
    }


    // FILE * neighbfunFile = fopen(neighbfunFileName.c_str(),"wb");
    // if (neighbfunFile == NULL){
    //     std::cerr << "The neighbors nodes file is not open" << std::endl;
    //     exit(1);
    // }



    FILE * edgesFile = fopen(edges.c_str(),"wb");
    if (edgesFile == NULL){
        std::cerr << "The edges file is not open" << std::endl;
        exit(1);
    }
    FILE * edgestypesFile = fopen(edgestypesFilename.c_str(),"wb");
    if (edgestypesFile == NULL){
        std::cerr << "The edges types file is not open" << std::endl;
        exit(1);
    }


    FILE * directionsFile = fopen(directions.c_str(),"wb");
    if (directionsFile == NULL){
        std::cerr << "The directions file is not open" << std::endl;
        exit(1);
    }


    FILE * scoreoutputFile = fopen(scoreoutputFilename.c_str(),"wb");
    if (scoreoutputFile == NULL){
      std::cerr << "The scores file is not open" << std::endl;
      exit(1);
    }
    //if (add_sph_harm_bn_nodes){
        FILE * sphNodesFile = fopen(sphNodesFilename.c_str(),"wb");
        if (sphNodesFile == NULL){
            std::cerr << "The  file with spherical harmonics of angles between nodes  is not open" << std::endl;
            exit(1);
        }  
    //}


    

    std::ifstream scoreFile;
	  std::string line;
    std::cout << scoreFilename << std::endl;
	  if (!scoreFilename.empty()) {
      
		    scoreFile.open(scoreFilename, std::ios_base::in);
	  }
    float score = -1;
    int resCode = -1;

    bool non_found;
    cVector3 local_coord, spherical_coord;
    int type;

    std::vector <std::pair<size_t, size_t>> res_indices;
    std::vector<std::string> tokens;
    std::string token;


    //first, compute all local frames
    for (auto &residue : protein->residues()) {
        if(residue.setLocalFrame()==0) {
            std::cerr << "No backbone atoms in " << residue.resName << " "<< residue.seqNumber <<"\n";
            return false;
        }
    }

    for (auto const &resi : protein->residues()) {
        // FIXME : should we compute all the atoms within certain radius, or only residue atoms?
        // here we compute expansions for the atomic representation
        // resi.origin.print();
        res_types.push_back(res_map[resi.resName]);
        sh.fillCoefficientsWithZeros();
        num_at = 0;
        for (auto const &atom : protein->atoms()) {
              // if ((atom.getPosition() - resi.origin).norm() < Radius){
              if ((resi.worldToLocal*atom.getPosition()).norm2() < Radius*Radius){
                  type = A.getType((cAtom *)&atom);
                  if (type >= 0){
                      // local_coord = resi.worldToLocal*(atom.getPosition() - resi.origin);
                      local_coord = resi.worldToLocal*atom.getPosition();
                      spherical_coord = sh.cart2Sph(local_coord);
                      // std::cout << (atom.getPosition() - resi.origin).norm() << std::endl;
                      // std::cout << local_coord.norm() << std::endl;
                      // std::cout << spherical_coord[0] << " " << spherical_coord[1] << " " << spherical_coord[2] << std::endl;
                      // std::cout << local_coord << " " << atom.name  << " " << atom.resName << std::endl; 
                      sh.computeCoefficients(spherical_coord, nFourierShells, maxQ, type, sigma);
                      num_at += 1;
                  }

                  
                  
                  
              }
              

        }
        
        if (add_solvent){
            num_sa = 0;
            for(auto const& solv_res: solvent_residues) {
                for(auto const& solv_atom: solv_res.get()->atoms()){
                    if ((resi.worldToLocal*solv_atom.getPosition()).norm() < Radius){
                        
                        local_coord = resi.worldToLocal*solv_atom.getPosition();
                        // std::cout << local_coord << " " << solv_atom.name  << " " << solv_atom.resName << std::endl; 
                        spherical_coord = sh.cart2Sph(local_coord);
                        sh.computeCoefficients(spherical_coord, nFourierShells, maxQ, 167, sigma);
                        num_sa += 1;
                    }
                }

            }
            // std::cout << num_at << "  " << num_sa << std::endl;
        }
        for (int i = 0; i < num_types; i++){
            for (int p = 0; p < nFourierShells; p++){
                for (int l = 0; l < order; l++){
                    for (int m = 0; m < l+1; m++){
                        Real x = sh.Coefficients[i][p][l][m].x;
                        Real y = sh.Coefficients[i][p][l][m].y;
                        fwrite(&x, sizeof(Real), 1, expansionsFile);
                        fwrite(&y, sizeof(Real), 1, expansionsFile);
                    }
                }
            }
        }
    }


   size_t res_index = 0;
  cVector3 dif;
  bool eof = false;

    for (auto &resi : protein->residues()) {
        for (auto &tar_resi : target_protein->residues()) {
            if (resi.seqNumber == tar_resi.seqNumber){
                dif = tar_resi.origin - resi.origin;
                if (!scoreFilename.empty()) {
                    
                      while (resCode < (int)resi.seqNumber && !eof && scoreFile) {
                          if (score_type == "cad"){
                              std::getline(scoreFile, line);
                              if (line.empty()) {
                                eof = true;
                                break;
                              }
                              int count = 0;
                              while (line.at(count) != 'r') {
                                count++;
                              }
                              assert(line.at(count) == 'r');
                              count++;
                              assert(line.at(count) == '<');
                              count++;
                              int intStart = count;
                              while (line.length() > count && line.at(count) != '>') {
                                count++;
                              }
                              resCode = stoi(line.substr(intStart, count - intStart));
                              assert(line.at(count) == '>');
                              count++;
                              if (line.at(count) != 'R'){
                                                                    count+=4;
                                                            }

                              assert(line.at(count) == 'R');
                              count++;
                              assert(line.at(count) == '<');
                              while (line.length() > count && line.at(count) != '>') {
                                                                    count++;
                                                            }
                              assert(line.at(count) == '>');
                              count++;
                              assert(line.at(count) == ' ');
                              count++;
                              score = stof(line.substr(count));
                        
                          }else if (score_type == "lddt"){
                              std::getline(scoreFile, line);
                              // std::cout << line << std::endl;
                              if (line.empty() ) {
                                  eof = true;
                                  break;
                              }
                              if (line.find("\t") == string::npos || line.find("Chain") != string::npos){
                                  continue;
                              } 
                              
                              std::istringstream iss(line);
                              std::getline(iss, token, '\t');
                              std::getline(iss, token, '\t');
                              std::getline(iss, token, '\t');
                              
                              resCode = std::stoi(token);
                              std::getline(iss, token, '\t');
                              std::getline(iss, token, '\t');
                              score = std::stof(token);

                          
                          }
                        }
                    
                }
                int scoreInt;
                // std::cout << resCode << " " << resi.seqNumber << std::endl; 
                if (native){
                  scoreInt = 1000000;
                }
                else if (resCode == (int)resi.seqNumber) {
                  scoreInt = 1000000 * score;
                }
                else {
                  scoreInt = -1000000;
                }
                if (scoreInt > 0){
                  
                  directions_vector.push_back(dif);
                  scores_vector.push_back(scoreInt);
                  res_indices.push_back(std::make_pair(res_index, resi.seqNumber));
                  neighbours.push_back(0);
                  recomputed_directions_vector.push_back(cVector3(0.0, 0.0, 0.0));
                  res_index+=1;
                  
                }
                break;
            }
        }
    }
    // std::cout << directions_vector.size() << "  " << res_indices.size()  << "  " << res_index << std::endl;

    
    int neighbors_number;
    

    //now, compute all vs all orientations
    printf("#printing SH for %lu residues and %d expansion order. In total %d x %lu values\n", protein->numResidues(), order, order*order, protein->numResidues()*(protein->numResidues()-1));
    for (auto &resi : protein->residues()) {
        // non_found = true;
        // for (auto &tar_resi : target_protein->residues()) {
        //     if (resi.seqNumber == tar_resi.seqNumber){
                
        //         directions_vector.push_back(tar_resi.origin - resi.origin);
        //         non_found = false;
        //         break;
        //     }
        // }
        // if (non_found) {
        //     break;
        // }
        // res_index++;

        current_resInd = resi.seqNumber;
        auto find1 = std::find_if( res_indices.begin(), res_indices.end(), if_target_res_exists );

        if (find1 == res_indices.end() ){
          continue;
        }
        

        for (auto &resj : protein->residues()) {

            if(&resi == &resj) { //diagonal case
                continue;
            }

            current_resInd = resj.seqNumber;
            auto find2 = std::find_if( res_indices.begin(), res_indices.end(), if_target_res_exists );

            if (find2 == res_indices.end() ){
                continue;
            }


            if(((resi.origin - resj.origin).norm2() < Radius2*Radius2)){
                // std::cout << find1->second << std::endl;
                // std::cout << find2->second << std::endl;
                if (add_sph_harm_bn_nodes){
                  sh.computeSpharmonic( sh.cart2Sph(resi.worldToLocal*resj.origin) );
                  sComplex ** const Y_temp = sh.getSH();
                  for (int l = 0; l < order; l++){
                    for (int m = 0; m < l+1; m++){
                      Real x = Y_temp[l][m].x;
                      Real y = Y_temp[l][m].y;
                      fwrite(&x, sizeof(Real), 1, sphNodesFile);
                      fwrite(&y, sizeof(Real), 1, sphNodesFile);
                    }
                  }
                  for (int s = 0; s<nFourierShells; s++) {
                    Real q = s*M_PI/(Radius2); // from 0 to maxQ
                    if (!cSH::computeSphBessel((resi.worldToLocal*resj.origin).norm() * q, SphBesPoly[s], order)){
                        for (int l = 0; l < order; l++){
                            Real b = std::pow(2/M_PI,1/2)*q*SphBesPoly[s][l];
                            fwrite(&b, sizeof(Real), 1, sphNodesFile);
                        }
                    }
                    else{
                      
                        return false;
                    }
                  }
                }
                edges_vector.push_back(find1->first);
                edges_vector.push_back(find2->first);
                neighbors_number = resi.seqNumber > resj.seqNumber ? resi.seqNumber - resj.seqNumber : resj.seqNumber - resi.seqNumber;
                if (neighbors_number >= resgap){
                    neighbors_number = resgap;
                }
                
                // if (std::abs(resi.seqNumber - resj.seqNumber) == 1){
                //     neighbors_sign = 1;
                // }
                if (res_types[find1->first]<20 && res_types[find2->first]<20){
                    edges_types.push_back(neighbors_number*401 + (20*res_types[find1->first]+res_types[find2->first]));
                    // std::cout << neighbors_number*401 + (20*res_types[find1->first]+res_types[find2->first])  << " " << neighbors_number << " "  << resi.seqNumber << " " <<resj.seqNumber << " "<< (20*res_types[find1->first]+res_types[find2->first])<< std::endl;
                }else{
                    edges_types.push_back(neighbors_number*401 + 400);
                    // std::cout << neighbors_number*401 + 400 << " " << neighbors_number << " " << resi.seqNumber << " "<< resj.seqNumber << " "<< std::endl;
                }
                
                neighbours[find1->first] = neighbours[find1->first] + 1;
                recomputed_directions_vector[find1->first] = recomputed_directions_vector[find1->first] + cVector3(resi.vX|directions_vector[find2->first], resi.vY|directions_vector[find2->first], resi.vZ|directions_vector[find2->first]);
                
            }else{
                continue;
            }

            cSpatialTransform jIni = resi.worldToLocal*resj.localToWorld;

            // if (use_neighborhood_function_coefs){
            //     cVector3 jtoi = jIni.position;
            //     spherical_coord = sh.cart2Sph(jtoi);
            //     sh.computeBasisFunctions(spherical_coord, nFourierShells, maxQ);
                
            //     for (int p = 0; p < nFourierShells; p++){
            //         for (int l = 0; l < order; l++){
            //             for (int m = 0; m < l+1; m++){
            //                 Real x = sh.Coefficients[0][p][l][m].x;
            //                 Real y = sh.Coefficients[0][p][l][m].y;
            //                 fwrite(&x, sizeof(Real), 1, neighbfunFile);
            //                 fwrite(&y, sizeof(Real), 1, neighbfunFile);
            //             }
            //         }
            //     }
        


            // }

            if (use_aggregation_tensors){
            

            if (use_bessel_matrices){

            // coordinates of j's atoms in the ith frame
            //cSpatialTransform jIni = resi.worldToLocal*resj.localToWorld;

            double zDisplacement = jIni.position.norm();

            cMatrix33 firstRotation;
            if (zDisplacement<1e-10) {
                firstRotation = cMatrix33::identity();
            } else {
                
                firstRotation.rotateToAxis(jIni.position); //     rotates the matrix to a given axis over Z and Y axes, first by alpha, then by beta

                firstRotation = firstRotation.getTranspose(); // this rotates pointDirection to the Z axis
            }
            cMatrix33 secondRotation = firstRotation.getTranspose();

            firstRotation = firstRotation*jIni.orientation;

            // now we need to do a combination of rotation - translation - rotation
            double alpha, beta, gamma;
            firstRotation.computeEulerDecompositionActiveZYZ(alpha, beta, gamma); // it gives  R=R(z,gamma)R(y,beta)R(z,alpha)

            
            double temp_sign = 1.0;
            if (resi.seqNumber > resj.seqNumber) temp_sign = -1.0;

            // alpha = 0.0*temp_sign;
            // beta  = 0.0*temp_sign;
            // beta  = (1-temp_sign)*M_PI_2;
            // gamma = 0.0*temp_sign;
            

            // if (resi.seqNumber > resj.seqNumber) beta = M_PI;
            // std::cout << alpha << "  " << beta << "  " << gamma << std::endl;


            cSH::makeWignerTheta(beta, WignerD, order); // beta, Y-rotation
            cSH::makeWignerPhase(alpha, WignerPhaseA, order); // alpha, gamma, Z-rotations
            cSH::makeWignerPhase(gamma, WignerPhaseG, order); // alpha, gamma, Z-rotations

            cSH::writeSlaterMatrices(order, WignerPhaseA, WignerD, WignerPhaseG, outputFirstSlaterMatricesFile);

            // EXAMPLE
            //then, we are ready to apply the first rotation,
            // we need to simply save the matrices
//             for (int s = 0; s < nFourierShells; s++) {
// //               cSH::Rotate(WignerPhaseA, WignerD, WignerPhaseG, rotatedAlm[s], referenceAlm[s], order);
//                 cSH::Rotate(WignerPhaseA, WignerD, WignerPhaseG, test[s], Alm[s], order);
//             }

            // now, the Bessel functions

            // zDisplacement = 0.4;
            //  zDisplacement = 0.50*temp_sign;
            // if (resi.seqNumber > resj.seqNumber) zDisplacement *= -1;

            for (int s = 0; s<nFourierShells; s++) {
                Real q = s*maxQ/(nFourierShells - 1); // from 0 to maxQ
                if (!cSH::computeSphBessel(zDisplacement * q, SphBesPoly[s], order)){
                    sh.writeBesselMatrices (SphBesPoly[s], order, order, outputBesselMatricesFile);
                }
                else{
                  
                    return false;
                }
                // std::cout << "d = " << zDisplacement * q << std::endl;
                // for (int p =0; p < order; p++){
                //   std::cout << "order " << p << ": " << SphBesPoly[s][p] << std::endl;
                // }
                
            }

            


            
            // EXAMPLE
            // translate teh coefficients
//             for (int s=0; s < nFourierShells; s++){
// //                sh.translateZ(translatedAlm[s], rotatedAlm[s], SphBesPoly[s]);
//                 sh.translateZ(test[s], Alm[s], SphBesPoly[s], order, order);
//             }


            secondRotation.computeEulerDecompositionActiveZYZ(alpha, beta, gamma); // it gives  R=R(z,gamma)R(y,beta)R(z,alpha)


            
            // alpha =  0.0*temp_sign;
            // beta  =  0.0*temp_sign;
            // beta  = (1-temp_sign)*M_PI_2;
            // gamma =  0.0*temp_sign;

            // if (resi.seqNumber > resj.seqNumber) beta = M_PI;
            // std::cout << alpha << "  " << beta << "  " << gamma << std::endl;


            cSH::makeWignerTheta(beta, WignerD, order); // beta, Y-rotation
            cSH::makeWignerPhase(alpha, WignerPhaseA, order); // alpha, gamma, Z-rotations
            cSH::makeWignerPhase(gamma, WignerPhaseG, order); // alpha, gamma, Z-rotations

            cSH::writeSlaterMatrices(order, WignerPhaseA, WignerD, WignerPhaseG, outputSecondSlaterMatricesFile);

            // EXAMPLE
//             for (int s = 0; s < nFourierShells; s++) {
// //                cSH::Rotate(WignerPhaseA, WignerD, WignerPhaseG, finalAlm[s], translatedAlm[s], P);
//                 cSH::Rotate(WignerPhaseA, WignerD, WignerPhaseG, test[s], Alm[s], order);
//             }
            }else{
                //cSpatialTransform jIni = resi.worldToLocal*resj.localToWorld;
                cMatrix33 firstRotation = jIni.orientation;
                double alpha, beta, gamma;
                firstRotation.computeEulerDecompositionActiveZYZ(alpha, beta, gamma);
                cSH::makeWignerTheta(beta, WignerD, order); // beta, Y-rotation
                cSH::makeWignerPhase(alpha, WignerPhaseA, order); // alpha, gamma, Z-rotations
                cSH::makeWignerPhase(gamma, WignerPhaseG, order); // alpha, gamma, Z-rotations

                cSH::writeSlaterMatrices(order, WignerPhaseA, WignerD, WignerPhaseG, outputFirstSlaterMatricesFile);





            }
            
            
            }

        }
        recomputed_directions_vector[find1->first] = -recomputed_directions_vector[find1->first]/neighbours[find1->first]+cVector3(resi.vX|directions_vector[find1->first], resi.vY|directions_vector[find1->first], resi.vZ|directions_vector[find1->first]);
    }
    size_t a1 = directions_vector.size();
    size_t a2 = edges_vector.size()/2;
    // std::cout << a1 << "  " << a2 << std::endl;
    // void* vd = &a;
    fwrite(&a1, sizeof(size_t), 1, generalFile);
    fwrite(&a2, sizeof(size_t), 1, generalFile);
    // fwrite(&a2, sizeof(size_t), 1, generalFile);



    // for (int n = 0; n < a1; n++){
    //     for (int i = 0; i < 167; i++){
    //         for (int p = 0; p < nFourierShells; p++){
    //             for (int l = 0; l < order; l++){
    //                 for (int m = 0; m < 2*l+2; m++){
    //                     double r = -1 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(2)));
    //                     fwrite(&r, sizeof(double), 1, expansionsFile);
    //                 }
    //             }
    //         }
    //     }
    // }

    // int j = 0;
    // for(size_t i = 0; i < edges_vector.size()/2; i++){ 
    //   fwrite(&i, sizeof(size_t),1, edgesFile );
    //   j += 1;
    // }
    
    // std::cout << j << std::endl;

    for (const auto &e: edges_vector){

      // j += -1;
      // std::cout << e << std::endl;
      fwrite(&e, sizeof(size_t),1, edgesFile );
    }

    for (const auto &e: edges_types){

      // j += -1;
      // std::cout << e << std::endl;
      fwrite(&e, sizeof(int),1, edgestypesFile );
    }

    // std::cout << j << std::endl;
    Real x = 0 , y = 0,  z = 0;
    std::cout << a1 << std::endl;
    for (size_t vec_i = 0 ; vec_i < a1; vec_i ++){
      
      cVector3 d = recomputed_directions_vector[vec_i];// - directions_vector[vec_i];
      // d.get(&x,&y,&z);
      x = (Real)d[0];
      y = (Real)d[1];
      z = (Real)d[2];
      // std::cout << x << " " << y << " " << z << std::endl;
      // std::cout << neighbours[vec_i] <<  std::endl;
      fwrite(&x, sizeof(Real),1, directionsFile );
      fwrite(&y, sizeof(Real),1, directionsFile );
      fwrite(&z, sizeof(Real),1, directionsFile );
      // fwrite(&x, sizeof(double),1, directionsFile );
      // fwrite(&y, sizeof(double),1, directionsFile );
      // fwrite(&z, sizeof(double),1, directionsFile );
      
      fwrite(&scores_vector[vec_i], sizeof(int), 1, scoreoutputFile);
    }
    
    // cMemoryManager::dellocTripleArray(Alm, nFourierShells, order);
    // cMemoryManager::dellocTripleArray(test, nFourierShells, order);
    cMemoryManager::dellocWignerRotation(WignerD, order);
    delete [] WignerPhaseA;
    delete [] WignerPhaseG;
    if (SphBesPoly) cMemoryManager::delloc2DArray(SphBesPoly);
    fclose(generalFile);
    fclose(expansionsFile);
    fclose(outputFirstSlaterMatricesFile);
    fclose(outputBesselMatricesFile);
    fclose(outputSecondSlaterMatricesFile);
    fclose(edgesFile);
    fclose(edgestypesFile);
    fclose(directionsFile);
    fclose(scoreoutputFile);
    //if (add_sph_harm_bn_nodes){
      fclose(sphNodesFile);
    //}
    // fclose(neighbfunFile);
    return true;

}



bool testBesselMatrices(cProtein *protein,  const std::string &expansions,  const std::string &outputBesselMatrices, Real Radius){
    cSH sh;
    int order;
    int nFourierShells;
    int type;
    cVector3 local_coord, spherical_coord;
    cAtomicResidueMapper A(167, 12, 1.0, 1.6);
    Real maxQ = 5.0;
    Real z_diff = 2.0;
    cVector3 z_shift(0.0,0.0,z_diff);
    // sComplex **** newCoefficients;
    for (order = 2; order < 100; order++){
        nFourierShells = order;
        sh.init(order);
        sh.initWigner(order, order);
        // cMemoryManager::allocQuadArray(newCoefficients, 167, order, order);
        FILE * expansionsFile1 = fopen((expansions + "_" + std::to_string(order) + "_init").c_str(),"wb");
        if (expansionsFile1 == NULL){
            std::cerr << "The nodes file is not open" << std::endl;
            exit(1);
        }
        FILE * expansionsFile2 = fopen((expansions + "_" + std::to_string(order) + "_shifted").c_str(),"wb");
        if (expansionsFile2 == NULL){
            std::cerr << "The nodes file is not open" << std::endl;
            exit(1);
        }
        // FILE * expansionsFile3 = fopen((expansions + "_" + std::to_string(order) + "_init_transform").c_str(),"wb");
        // if (expansionsFile3 == NULL){
        //     std::cerr << "The nodes file is not open" << std::endl;
        //     exit(1);
        // }
        // FILE * expansionsFile4 = fopen((expansions + "_" + std::to_string(order) + "_shifted_transform").c_str(),"wb");
        // if (expansionsFile4 == NULL){
        //     std::cerr << "The nodes file is not open" << std::endl;
        //     exit(1);
        // }
        FILE * outputBesselMatricesFile1 = fopen((outputBesselMatrices + "_" + std::to_string(order) + "_to").c_str(),"wb");
        if (outputBesselMatricesFile1 == NULL){
            std::cerr << "The bm file is not open" << std::endl;
            exit(1);
        }
        FILE * outputBesselMatricesFile2 = fopen((outputBesselMatrices + "_" + std::to_string(order) + "_back").c_str(),"wb");
        if (outputBesselMatricesFile2 == NULL){
            std::cerr << "The bm file is not open" << std::endl;
            exit(1);
        }
        Real    **SphBesPoly;
        cMemoryManager::alloc2DArray(SphBesPoly, nFourierShells, order);

        for (auto &residue : protein->residues()) {
          if(residue.setLocalFrame()==0) {
              std::cerr << "No backbone atoms in " << residue.resName << " "<< residue.seqNumber <<"\n";
              return false;
          }
        }
        

        auto resi = protein->residues().begin(); 
        {
            sh.fillCoefficientsWithZeros();
            for (auto const &atom : protein->atoms()) {
                  if ((resi->worldToLocal*atom.getPosition()).norm() < Radius){
                      type = A.getType((cAtom *)&atom);
                      if (type >= 0){
                        local_coord = resi->worldToLocal*atom.getPosition();
                        spherical_coord = sh.cart2Sph(local_coord);
                        sh.computeCoefficients(spherical_coord, nFourierShells, maxQ, type);
                      }
                      
                      
                  }
                  

            }
            int full_size = 0;
            for (int i = 0; i < 167; i++){
                for (int p = 0; p < nFourierShells; p++){
                    for (int l = 0; l < order; l++){
                        for (int m = 0; m < l+1; m++){
                            Real x = sh.Coefficients[i][p][l][m].x;
                            Real y = sh.Coefficients[i][p][l][m].y;
                            fwrite(&x, sizeof(Real), 1, expansionsFile1);
                            fwrite(&y, sizeof(Real), 1, expansionsFile1);
                            full_size += 2*sizeof(Real);
                        }
                    }
                }
            }
            // std::cout << full_size << std::endl;
            // std::cout << 4*167*nFourierShells*(order+1)*order << std::endl;
            sh.fillCoefficientsWithZeros();
            for (auto const &atom : protein->atoms()) {
                  if ((resi->worldToLocal*atom.getPosition()).norm() < Radius){
                      type = A.getType((cAtom *)&atom);
                      if (type >= 0){
                        local_coord = resi->worldToLocal*atom.getPosition() - z_shift;
                        spherical_coord = sh.cart2Sph(local_coord);
                        sh.computeCoefficients(spherical_coord, nFourierShells, maxQ, type);
                      }
                      
                      
                  }
                  

            }
            full_size = 0;
            for (int i = 0; i < 167; i++){
                for (int p = 0; p < nFourierShells; p++){
                    for (int l = 0; l < order; l++){
                        for (int m = 0; m < l+1; m++){
                            Real x = sh.Coefficients[i][p][l][m].x;
                            Real y = sh.Coefficients[i][p][l][m].y;
                            fwrite(&x, sizeof(Real), 1, expansionsFile2);
                            fwrite(&y, sizeof(Real), 1, expansionsFile2);
                            full_size += 2*sizeof(Real);
                        }
                    }
                }
            }
            // std::cout << full_size << std::endl;
            // std::cout << 4*167*nFourierShells*(order+1)*order << std::endl;
        }
        int temp;
        int full_size = 0;
        for (int s = 0; s<nFourierShells; s++) {
            Real q = s*maxQ/(nFourierShells - 1); 
            if (!cSH::computeSphBessel(z_diff * q, SphBesPoly[s], order)){
                temp = sh.writeBesselMatrices (SphBesPoly[s], order, order, outputBesselMatricesFile1);
                full_size += temp;
                
            }
            else{
                  
                return false;
            }
        }
        // std::cout << full_size << std::endl;
        // std::cout << 4*nFourierShells*(order+1)*order*(2*order+1)/3 << std::endl;
        full_size = 0;
        // for (int i = 0; i < 167; i++){
        //         for (int p = 0; p < nFourierShells; p++){
        //             for (int l = 0; l < order; l++){
        //                 for (int m = 0; m < l+1; m++){
        //                     newCoefficients[i][p][l][m].x = 0.0;
        //                     newCoefficients[i][p][l][m].y= 0.0;
                            
        //                 }
        //             }
        //         }
        //     }
        
        for (int s = 0; s<nFourierShells; s++) {
            Real q = s*maxQ/(nFourierShells - 1); 
            if (!cSH::computeSphBessel(-z_diff * q, SphBesPoly[s], order)){
                temp = sh.writeBesselMatrices (SphBesPoly[s], order, order, outputBesselMatricesFile2);
                full_size += temp;
                // for (int f = 0; f < 167; f++)
                //     sh.translateZ(newCoefficients[f][s], sh.Coefficients[f][s],  SphBesPoly[s], order,order);
            }
            else{
                  
                return false;
            }
        }
        // for (int i = 0; i < 167; i++){
        //         for (int p = 0; p < nFourierShells; p++){
        //             for (int l = 0; l < order; l++){
        //                 for (int m = 0; m < l+1; m++){
        //                     Real x = newCoefficients[i][p][l][m].x;
        //                     Real y = newCoefficients[i][p][l][m].y;
        //                     fwrite(&x, sizeof(Real), 1, expansionsFile4);
        //                     fwrite(&y, sizeof(Real), 1, expansionsFile4);
                            
        //                 }
        //             }
        //         }
        //     }

        // std::cout << full_size << std::endl;
        // std::cout << 4*nFourierShells*(order+1)*order*(2*order+1)/3 << std::endl;
        if (SphBesPoly) cMemoryManager::delloc2DArray(SphBesPoly);
        // if (newCoefficients) cMemoryManager::dellocQuadArray(newCoefficients, 167, order, order);
        fclose(expansionsFile1);
        fclose(expansionsFile2);
        // fclose(expansionsFile3);
        // fclose(expansionsFile4);
        fclose(outputBesselMatricesFile1);
        fclose(outputBesselMatricesFile2);


    }

    
    return true;

}

bool writeSHfeatures(cProtein *protein, const int order, bool orth) {

    //allocate memory + precompute constants
    cSH sh;
    sh.init(order);

    //first, compute all local frames
    for (auto &residue : protein->residues()) {
        if(residue.setLocalFrame()==0) {
            std::cerr << "No backbone atoms in " << residue.resName << " "<< residue.seqNumber <<"\n";
            return false;
        }
    }

    //now, compute all vs all orientations
    printf("#printing SH for %lu residues and %d expansion order. In total %d x %lu values\n", protein->numResidues(), order, order*order, protein->numResidues()*(protein->numResidues()-1));
    for (auto &resi : protein->residues()) {
        for (auto &resj : protein->residues()) {

            if(&resi == &resj) { //diagonal case
                continue;
            }

            cVector3 worldJ = resj.getAtom("CA")->getPosition();
            cVector3 jInI = resi.worldToLocal*worldJ;

            cVector3 localSphCoords = cSH::cart2Sph(jInI);
            sh.computeSpharmonic(localSphCoords);

            //OK, now, SH coeeficients are in Y_C[n][m].x and Y_C[n][m].y!
            //We have order^2 values
            if (orth == false) {
            for (int l=0; l < order; l++) {
                for (int m=1; m <= l; m++) {
                    printf("%8.4f", -2*sh.getSH()[l][m].y); // the -m part
                }

                printf("%8.4f", sh.getSH()[l][0].x);

                for (int m=1; m <= l; m++) {
                    printf("%8.4f", +2*sh.getSH()[l][m].x); // the +m part
                }
            }
            } else { //orthonormal version
                for (int l=0; l < order; l++) {
                    for (int m=1; m <= l; m++) {
                        printf("%8.4f", pow(-1,m)*sqrt(2)*sh.getSH()[l][m].y); // the -m part = -pow(-1,m)/ sqrt(2) W[l][m]
                    }

                    printf("%8.4f", sh.getSH()[l][0].x); // =  W[l][m]

                    for (int m=1; m <= l; m++) {
                        printf("%8.4f", pow(-1,m)*sqrt(2)*sh.getSH()[l][m].x); // the +m part = pow(-1,m)/ sqrt(2) W[l][m]
                    }
                }

            }
            printf("\n");
        }
    }

    return true;
}


bool getMatricesLinTransform(const std::string &first_slater_matrices,  const std::string &bessel_matrices,  const std::string &second_slater_matrices,  size_t order, double radius, double maxq,  double alpha1, double beta1, double gamma1, double zshift, double alpha2, double beta2, double gamma2){
    FILE * outputFirstSlaterMatricesFile = fopen(first_slater_matrices.c_str(),"wb");
    if (outputFirstSlaterMatricesFile == NULL){
        std::cerr << "The fsm file is not open" << std::endl;
        exit(1);
    }
    FILE * outputBesselMatricesFile = fopen(bessel_matrices.c_str(),"wb");
    if (outputBesselMatricesFile == NULL){
        std::cerr << "The bm file is not open" << std::endl;
        exit(1);
    }
    FILE * outputSecondSlaterMatricesFile = fopen(second_slater_matrices.c_str(),"wb");
    if (outputSecondSlaterMatricesFile == NULL){
        std::cerr << "The ssm file is not open" << std::endl;
        exit(1);
    }
    cSH sh;
    sh.init(order);
    sh.initWigner(order, order);
    sComplex *WignerPhaseA = new sComplex[order];
    sComplex *WignerPhaseG = new sComplex[order];
    Real        ***WignerD;
    cMemoryManager::allocWignerRotation(WignerD, order);
    Real    **SphBesPoly;
    cMemoryManager::alloc2DArray(SphBesPoly, order, order);
    cSH::makeWignerTheta(beta1, WignerD, order); // beta, Y-rotation
    cSH::makeWignerPhase(alpha1, WignerPhaseA, order); // alpha, gamma, Z-rotations
    cSH::makeWignerPhase(gamma1, WignerPhaseG, order); // alpha, gamma, Z-rotations

    cSH::writeSlaterMatrices(order, WignerPhaseA, WignerD, WignerPhaseG, outputFirstSlaterMatricesFile);
    for (int s = 0; s<order; s++) {
        Real q = s*maxq/(order - 1); // from 0 to maxQ
        if (!cSH::computeSphBessel(zshift * q, SphBesPoly[s], order)){
            sh.writeBesselMatrices (SphBesPoly[s], order, order, outputBesselMatricesFile);
        }
        else{
                  
            return false;
        }
    }
    cSH::makeWignerTheta(beta2, WignerD, order); // beta, Y-rotation
    cSH::makeWignerPhase(alpha2, WignerPhaseA, order); // alpha, gamma, Z-rotations
    cSH::makeWignerPhase(gamma2, WignerPhaseG, order); // alpha, gamma, Z-rotations

    cSH::writeSlaterMatrices(order, WignerPhaseA, WignerD, WignerPhaseG, outputSecondSlaterMatricesFile);
    fclose(outputFirstSlaterMatricesFile);
    fclose(outputBesselMatricesFile);
    fclose(outputSecondSlaterMatricesFile);


}


bool correlationProteins(cProtein *protein1, cProtein *protein2, const std::string &outputFilename){
    Real Rmax1, Rmax2, Rmax, deltaQ;
    double *mass1, *mass2;
    sComplex ***outputStruct;
    Real x, y;
    cVector3 cm1, cm2;
    cVector3 local_coord, spherical_coord;
    cSH sh;
    std::clock_t time0, time1;
    cm1 = protein1->getCM(mass1);
    cm2 = protein2->getCM(mass2);
    Rmax1 = (Real)protein1->getMaxR(cm1);
    Rmax2 = (Real)protein2->getMaxR(cm2);
    Rmax = std::max(Rmax1, Rmax2);
    deltaQ = M_PI/Rmax;
    int L, L_max;
    L_max = 100;
    cMemoryManager::allocTripleArray(outputStruct, L_max, L_max);
    sh.initConv6D(L_max);
    sh.initWignerConv6D(L_max);
    for (const auto &atom : protein1->atoms()){
        local_coord = atom.getPosition() - cm1;
        spherical_coord = sh.cart2Sph(local_coord);
        sh.computeCoefficientsTripleArray(sh.FirstStruct, spherical_coord, L_max, deltaQ, Rmax/(L_max-1));
    }
    for (const auto &atom : protein2->atoms()){
        local_coord = atom.getPosition() - cm2;
        spherical_coord = sh.cart2Sph(local_coord);
        sh.computeCoefficientsTripleArray(sh.SecondStruct, spherical_coord, L_max, deltaQ, Rmax/(L_max-1));
    }

    FILE * outputFile = fopen((outputFilename).c_str()  ,"wb");
    if (outputFile == NULL){
        std::cerr << "The output file is not open. " << std::endl;
        exit(1);
    }
    
    for(L = 5; L <= L_max; L+=5){
        time0 = std::clock();
        
        sh.Convolution6D(sh.FirstStruct, sh.SecondStruct, outputStruct, L, L, L_max);
        

        for (int p = 0; p < L; p++){
            for (int l = 0; l < L; l++){
                for (int m = 0; m < l+1; m++){
                    x = outputStruct[p][l][m].x;
                    y = outputStruct[p][l][m].y;
                    outputStruct[p][l][m].x = 0.0;
                    outputStruct[p][l][m].y = 0.0;
                    //fwrite(&x, sizeof(Real), 1, outputFile);
                    //fwrite(&y, sizeof(Real), 1, outputFile);
                }
            }
        }


        // fclose(outputFile);
        time1 = std::clock();
        std::cout << "L:" << L<<":" <<1000*float(time1-time0)/CLOCKS_PER_SEC <<std::endl;

    }

    fclose(outputFile);

}

bool writeAllMaps(cProtein *protein, cProteinMapper *mapper, const std::string &outputFilename, bool native, const std::string &scoreFilename, std::map<int, char> ssByRes, std::map<int, float> areaByRes) {
	std::fstream binOutputFile;
	mapper->setProtein(protein);
	binOutputFile.open(outputFilename, std::ios_base::out | std::ios_base::binary);
	FILE* binFile = fopen(outputFilename.c_str(),"wb");
	int nbMaps = mapper->getNbMaps(native, scoreFilename);
	int mapSize = mapper->getMapSize();
	int metaSize = mapper->getMetaSize();
	int* header = mapper->getHeader();
	header[7] = nbMaps;
	//binOutputFile.write((char*) header, header[0]);
	fwrite(header, sizeof(char), header[0], binFile);
	std::ifstream scoreFile;
	std::string line;
	if (!scoreFilename.empty()) {
		scoreFile.open(scoreFilename, std::ios_base::in);
	}

	int resCode = -1;
	float score = -1;
	bool eof = false;
	int nbMappedRes = 0;
	
		for (int indexMap = 0; indexMap < protein->numResidues(); indexMap++) {
			mapper->runCurrent();
			const uint8_t* map = mapper->getMap();
			int32_t* meta = mapper->getMeta();
			if (!scoreFilename.empty()) {
				while (resCode < ((int*)meta)[0] && !eof && scoreFile) {
					std::getline(scoreFile, line);
					if (line.empty()) {
						eof = true;
						break;
					}
					int count = 0;
					while (line.at(count) != 'r') {
						count++;
					}
					assert(line.at(count) == 'r');
					count++;
					assert(line.at(count) == '<');
					count++;
					int intStart = count;
					while (line.length() > count && line.at(count) != '>') {
						count++;
					}
					resCode = stoi(line.substr(intStart, count - intStart));
					assert(line.at(count) == '>');
					count++;
					if (line.at(count) != 'R'){
                                                count+=4;
                                        }

					assert(line.at(count) == 'R');
					count++;
					assert(line.at(count) == '<');
					while (line.length() > count && line.at(count) != '>') {
                                                count++;
                                        }
					assert(line.at(count) == '>');
					count++;
					assert(line.at(count) == ' ');
					count++;
					score = stof(line.substr(count));
				}
			}
			int scoreInt;
			if (native) {
				scoreInt = 1000000;
			}
			else if (resCode == ((int*)meta)[0]) {
				scoreInt = 1000000 * score;
			}
			else {
				scoreInt = -1000000;
			}
			meta[2] = scoreInt;
			if (ssByRes.find(((int*)meta)[0]) != ssByRes.end()) {
				meta[3] = ssByRes.at(((int*)meta)[0]);
				meta[4] = 10 * areaByRes.at(((int*)meta)[0]);
			}
			if (scoreInt >= 0) {
				//bool wm = writeMap(binOutputFile, map, meta, mapSize, metaSize);
				bool wm = writeMap(binFile, map, meta, mapSize, metaSize);
				
				if (!wm) return wm;
			}
			mapper->next();
			nbMappedRes++;
		}
		
		//std::cout << "Nb mapped residues : " << nbMappedRes << std::endl;
	return true;
}


bool computeGrad(cProtein *protein, cProteinMapper *mapper, const std::string &gradientMap, const std::string &outputFilename) {
	std::unordered_map<size_t, cVector3> gradByAtom;
	for (cAtom at : protein->atoms()) {
		gradByAtom.insert({ {at.atomSerial, cVector3(0, 0, 0)} });
	}
	protein->complete(true);
	mapper->setProtein(protein);

	int nbMaps = mapper->getNbMaps(true, "");
	int mapSize = mapper->getMapSize();
	std::vector<unsigned long> retypeMatShape;
	std::vector<float> retypeMat;

	std::ifstream stream(gradientMap, std::ifstream::binary);
	if (!stream) {
		throw std::runtime_error("io error: failed to open a file.");
	}
	{
		std::string header = npy::read_header(stream);
		// parse header
		bool fortran_order;
		std::string typestr;
		npy::parse_header(header, typestr, fortran_order, retypeMatShape);

		// check if the typestring matches the given one
		npy::Typestring typestring_o{ retypeMat };
		std::string expect_typestr = typestring_o.str();
		if (typestr != expect_typestr) {
			throw std::runtime_error("formatting error: typestrings not matching");
		}

		// compute the data size based on the shape
		auto size = static_cast<size_t>(npy::comp_size(retypeMatShape));
		retypeMat.resize(size);

		// read the data
		stream.read(reinterpret_cast<char*>(retypeMat.data()), sizeof(float)*size);
	}

	mapper->numRetype = retypeMatShape[1];

	int resCode = -1;
	float score = -1;
	bool eof = false;
	for (int indexMap = 0; indexMap < protein->numResidues(); indexMap++) {
		std::vector<float> gradArray;
		std::vector<unsigned long> gradShape;
		{
			std::string header = npy::read_header(stream);
			// parse header
			bool fortran_order;
			std::string typestr;
			npy::parse_header(header, typestr, fortran_order, gradShape);

			// check if the typestring matches the given one
			npy::Typestring typestring_o{ gradArray };
			std::string expect_typestr = typestring_o.str();
			if (typestr != expect_typestr) {
				throw std::runtime_error("formatting error: typestrings not matching");
			}

			// compute the data size based on the shape
			auto size = static_cast<size_t>(npy::comp_size(gradShape));
			gradArray.resize(size);

			// read the data
			stream.read(reinterpret_cast<char*>(gradArray.data()), sizeof(float)*size);
		}
		mapper->runGradient(gradArray.data(),retypeMat.data(), gradByAtom);
		
		mapper->next();
	}


	cVector3 sumGradAll(0, 0, 0);


	FILE* f = fopen(outputFilename.c_str(), "w");
	int n = 0;
	for (cAtom at : protein->atoms()) {
		n++;
		if (gradByAtom.find(at.atomSerial) != gradByAtom.end()) {
			at.setPosition(gradByAtom[at.atomSerial]);
			sumGradAll += gradByAtom[at.atomSerial];
			gradByAtom.erase(at.atomSerial);
			at.toFile(f);
		}
		else {

			at.setPosition(cVector3(0,0,0));
			at.toFile(f);
		}
	}

	return true;
}


bool writeFeatureVector(cProtein *protein,
                        cProteinFeaturizer *featurizer,
                        const std::string &filename) {
  auto feature_vector = featurizer->featurize(protein);
  return write_vector(feature_vector, filename);
}

void greedySearch(cProtein *protein, size_t numIterations) {
  printf("Iteration     Energy           Changed rotamers\n");

  vector<size_t> rotamers;
  for (auto &residue : protein->residues()) {
    cAminoResidue *res = dynamic_cast<cAminoResidue *>(&residue);
    if (!res || res->numRotamers() < 2)
      continue;

    res->setRotamer(0);
    rotamers.push_back(0);
  }

  energy::initializeNeighbourAtoms(protein);
  for (size_t k = 0; k < numIterations; ++k) {

    double totalEnergySum = 0;
    size_t numChangedRotamers = 0;
    size_t idx = 0;
    for (auto &residue : protein->residues()) {
      cAminoResidue *res = dynamic_cast<cAminoResidue *>(&residue);
      if (!res || res->numRotamers() < 2)
        continue;

      double minEnergy = 1E20;
      size_t bestRotamer = -1;
      double libEnergy;

      for (size_t j = 0; j < res->numRotamers() && j < kMaxRotamersPerResidue; ++j) {
        res->setRotamer(j, &libEnergy);
        double energy = energy::totalEnergy(*res);
        energy += libEnergy;
        if (energy < minEnergy) {
          minEnergy = energy;
          bestRotamer = j;
        }
      }
      assert(bestRotamer != static_cast<size_t>(-1));
      if (bestRotamer != rotamers[idx]) {
        numChangedRotamers++;
        rotamers[idx] = bestRotamer;
      }
      res->setRotamer(bestRotamer);
      totalEnergySum += minEnergy;
      idx++;
    }
    printf("%9zd     %6lf          %zd\n", k, totalEnergySum, numChangedRotamers);
    if (!numChangedRotamers)
      break;
  }
#if 0
  size_t idx = 0;
  for (residueIterator resIt = protein->residuesBegin();
                        resIt != protein->residuesEnd(); ++resIt) {
    cAminoResidue *res = dynamic_cast<cAminoResidue *>(&(*resIt));
    if (!res || res->numRotamers() < 2)
      continue;
    for (size_t i = 0; i < res->numRotamers() && i < kMaxRotamersPerResidue; ++i) {
      if (i == rotamers[idx]) {
        printf("1 ");
      } else {
        printf("0 ");
      }
    }
    idx++;
  }
  printf("\n%d\n", rotamers.size());
#endif
}

void doSmth(cProtein *protein) {
  // writeEnergyMatrix("Energy");
  // setRotamers("vector.txt");

	// some output
#if 1
  greedySearch(protein, kNumIterations);
#else
  energy::initializeNeighbourAtoms(protein);

  for (size_t k = 1; k <= kNumIterations; ++k) {

    for (residueIterator resIt = protein->residuesBegin();
                          resIt != protein->residuesEnd(); ++resIt) {
      cAminoResidue *res = dynamic_cast<cAminoResidue *>(&(*resIt));
      if (!res || !res->numRotamers())
        continue;

      double min_energy = 1000000;
      double initial_energy = energy::totalEnergy(*resIt);
      double libEnergy;
      size_t best_rotamer;
      for (size_t j = 0; j < res->numRotamers(); ++j) {
        res->setRotamer(j, &libEnergy);
        double energy = energy::totalEnergy(*res);
        energy += libEnergy;
        if (energy < min_energy) {
          min_energy = energy;
          best_rotamer = j;
        }
      }
      double prob = res->setRotamer(best_rotamer, &libEnergy);
      if (k == kNumIterations) {
        std::cout << "Chain " << res->chainId << ": Residue "
                  << res->seqNumber << ": phi " << res->getPhi()
                  << ", psi " << res->getPsi()
                  << " is set to " << best_rotamer
                  << " rotamer with probability: " << prob
                  << " Total: " << min_energy
                  << " Initial: " << initial_energy
                  << " Relative: " << libEnergy << std::endl;
      }
    }
  }
#endif
/*
  cChain::atomIterator atom = chains[0]->atomsBegin();
  for (set<cAtom *>::iterator it = atom->neighbours.begin();
                                     it != atom->neighbours.end(); ++it)
    std::cout << *it << std::endl;
*/
}

bool writeEnergyMatrix(cProtein *protein, const std::string &filename) {
  vector<cAminoResidue *> residues;
  vector<vector<cAminoResidue *>> neighbours
      = energy::initializeNeighbourResidues(protein, &residues);
  // reindexing residues and counting number of all rotamers
  size_t numRotamers = 0;
  for (size_t resIdx = 0; resIdx < residues.size(); ++resIdx) {
    residues[resIdx]->residueIndex = numRotamers;
    numRotamers += std::min(residues[resIdx]->numRotamers(),
                            kMaxRotamersPerResidue);
  }
#if 0
  cSparseMatrixOutputASCII matrixOutput;
#else
  cSparseMatrixOutputMAT matrixOutput(numRotamers, numRotamers + 3);
#endif
  if (!matrixOutput.initialize(filename))
    return false;

  // Write inexes of all rotamers
  // residues
  for (size_t resIdx = 0; resIdx < residues.size(); ++resIdx) {
    cAminoResidue &residue = *residues[resIdx];

    // rotamers
    for (size_t rotIdx = 0; rotIdx < kMaxRotamersPerResidue
                            && rotIdx < residue.numRotamers(); ++rotIdx) {

      if (!matrixOutput.writeTriplet(residue.residueIndex + rotIdx, 0,
                                      residue.residueIndex))
        return false;
    }
  }
  // Write libEnergy
  // residues
  for (size_t resIdx = 0; resIdx < residues.size(); ++resIdx) {
    cAminoResidue &residue = *residues[resIdx];

    // rotamers
    for (size_t rotIdx = 0; rotIdx < kMaxRotamersPerResidue
                            && rotIdx < residue.numRotamers(); ++rotIdx) {

      double libEnergy;
      residue.setRotamer(rotIdx, &libEnergy);
      if (!matrixOutput.writeTriplet(residue.residueIndex + rotIdx, 1,
                                      libEnergy))
        return false;
    }
  }
  // Write frameEnergy
  // residues
  for (size_t resIdx = 0; resIdx < residues.size(); ++resIdx) {
    cAminoResidue &residue = *residues[resIdx];

    // rotamers
    for (size_t rotIdx = 0; rotIdx < kMaxRotamersPerResidue
                            && rotIdx < residue.numRotamers(); ++rotIdx) {
      residue.setRotamer(rotIdx);
      if (!matrixOutput.writeTriplet(residue.residueIndex + rotIdx, 2,
                                      energy::frameEnergy(residue)))
        return false;
    }
  }
  // Write pairEnergy
  // residues
  for (size_t resIdx = 0; resIdx < residues.size(); ++resIdx) {
    cAminoResidue &residue = *residues[resIdx];

    // rotamers
    for (size_t rotIdx_1 = 0; rotIdx_1 < kMaxRotamersPerResidue
                              && rotIdx_1 < residue.numRotamers(); ++rotIdx_1) {
      residue.setRotamer(rotIdx_1);
      if (!matrixOutput.writeTriplet(residue.residueIndex + rotIdx_1,
                                      3 + residue.residueIndex + rotIdx_1,
                                      energy::pairEnergy(residue, residue) / 2))
        return false;
      // neighbour residues
      for (size_t nbIdx = 0; nbIdx < neighbours[resIdx].size(); ++nbIdx) {
        cAminoResidue &neighbour = *neighbours[resIdx][nbIdx];

        // rotamers of neighbour residue
        for (size_t rotIdx_2 = 0; rotIdx_2 < kMaxRotamersPerResidue
                                  && rotIdx_2 < neighbour.numRotamers(); ++rotIdx_2) {

          neighbour.setRotamer(rotIdx_2);
          if (!matrixOutput.writeTriplet(neighbour.residueIndex + rotIdx_2,
                                          3 + residue.residueIndex + rotIdx_1,
                                          energy::pairEnergy(residue, neighbour)))
            return false;
        }
      }
    }
  }
  return matrixOutput.deinitialize();
}

inline bool areAnglesEqual(double angle_1, double angle_2, double dev) {
  double diff = std::abs(angle_1 - angle_2);
  return std::abs(diff) < dev || std::abs(diff - 360.0) < dev;
}

double chi1Quality(const cProtein &protein, const cProtein &ethalon) {
  size_t numAngles = 0;
  size_t numEqualAngles = 0;

  for (auto resIt = protein.residues().begin(), ethalonResIt = ethalon.residues().begin();
            resIt != protein.residues().end() && ethalonResIt != ethalon.residues().end();
                ++resIt, ++ethalonResIt) {
    const cAminoResidue *res
      = dynamic_cast<const cAminoResidue *>(&(*resIt));
    const cAminoResidue *ethalonRes
      = dynamic_cast<const cAminoResidue *>(&(*ethalonResIt));
    if (!res)
      continue;
    numAngles++;
    numEqualAngles += areAnglesEqual(res->getChi(1), ethalonRes->getChi(1), 40);
  }
  return static_cast<double>(numEqualAngles) / numAngles * 100;
}

double chi12Quality(const cProtein &protein, const cProtein &ethalon) {
  size_t numAngles = 0;
  size_t numEqualAngles = 0;

  for (auto resIt = protein.residues().begin(), ethalonResIt = ethalon.residues().begin();
            resIt != protein.residues().end() && ethalonResIt != ethalon.residues().end();
                ++resIt, ++ethalonResIt) {
    const cAminoResidue *res
      = dynamic_cast<const cAminoResidue *>(&(*resIt));
    const cAminoResidue *ethalonRes
      = dynamic_cast<const cAminoResidue *>(&(*ethalonResIt));
    if (!res)
      continue;
    numAngles++;
    numEqualAngles += areAnglesEqual(res->getChi(1), ethalonRes->getChi(1), 40)
                      * areAnglesEqual(res->getChi(2), ethalonRes->getChi(2), 40);
  }
  return static_cast<double>(numEqualAngles) / numAngles * 100;
}

double rmsdQuality(const cProtein &protein, const cProtein &ethalon) {
  double squareSum = 0.0;
  size_t numAtoms = 0;

  for (auto resIt = protein.residues().begin(),
            ethalonResIt = ethalon.residues().begin();
                resIt != protein.residues().end() &&
                ethalonResIt != ethalon.residues().end();
                    ++resIt, ++ethalonResIt) {
    for (const auto &first_atom : resIt->atoms()) {
      for (const auto &second_atom : ethalonResIt->atoms()) {
        if (!strcmp(first_atom.name, second_atom.name)) {
          numAtoms++;
          squareSum += (first_atom.getPosition() - second_atom.getPosition()).norm2();
        }
      }
    }
  }
  return sqrt(squareSum / numAtoms);
}

size_t clashQuality(const cProtein &protein) {
  return energy::numClashes(protein);
}

double energyQuality(cProtein *protein) {
  double total_energy = 0;

  energy::initializeNeighbourAtoms(protein);

  for (const auto &residue : protein->residues()) {
    if (const auto *amino_residue = dynamic_cast<const cAminoResidue *>(&residue))
      total_energy += energy::totalEnergy(*amino_residue);
  }
  return total_energy / 2;
}
