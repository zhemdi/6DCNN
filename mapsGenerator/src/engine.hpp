/*************************************************************************\

Sergei Grudinin, 2012
Mikhail Karasikov, 2016
All Rights Reserved.

\**************************************************************************/


#pragma once
#include <string>

#include "cProtein.hpp"
#include "cProteinFeaturizer.hpp"
#include "cProteinMapper.hpp"
#include "cSH.hpp"


bool writeFeatureVector(cProtein *protein,
                        cProteinFeaturizer *featurizer,
                        const std::string &);

bool writeEnergyMatrix(cProtein *protein, const std::string &);

bool writeSHfeatures(cProtein *protein, const int order, bool orth = false);

// bool writeAtomFeatures(cProtein *protein, cProtein *target_protein, const int order, const std::string &general, const std::string &expansions, const std::string &outputFirstSlaterMatrices, const std::string &outputBesselMatrices, const std::string &outputSecondSlaterMatrices, const std::string &neighbfunFileName, const std::string &edges, const std::string &directions, const std::string &scoreoutputFilename, const std::string &edgestypesFilename, const std::string &scoreFilename, const std::string &sphNodesFilename, const std::string &score_type, bool native,  Real Radius = 10,  Real Radius2 = 12,  Real maxQ = 1, Real sigma =1, bool orth = false, bool add_solvent = false, int resgap = 2, bool add_sph_harm_bn_nodes = false, bool use_aggregation_tensors = false, bool use_bessel_matrices = false, bool use_neighborhood_function_coefs = false);
bool writeAtomFeatures(cProtein *protein, cProtein *target_protein, const int order, const std::string &general, const std::string &expansions, const std::string &outputFirstSlaterMatrices, const std::string &outputBesselMatrices, const std::string &outputSecondSlaterMatrices, const std::string &edges, const std::string &directions, const std::string &scoreoutputFilename, const std::string &edgestypesFilename, const std::string &scoreFilename, const std::string &sphNodesFilename, const std::string &score_type, bool native,  Real Radius = 10,  Real Radius2 = 12,  Real maxQ = 1, Real sigma =1, bool orth = false, bool add_solvent = false, int resgap = 2, bool add_sph_harm_bn_nodes = false, bool use_aggregation_tensors = false, bool use_bessel_matrices = false);

bool writeAllMaps(cProtein *protein, cProteinMapper *mapper, const std::string &outputFilename, bool native, const std::string &scoreFilename , std::map<int, char> ssByRes, std::map<int, float> areaByRes);

bool computeGrad(cProtein *protein, cProteinMapper *mapper, const std::string &gradientMap, const std::string &outputFilename);

void greedySearch(cProtein *protein, size_t numIterations);

bool testBesselMatrices(cProtein *protein,  const std::string &expansions,  const std::string &outputBesselMatrices, Real Radius);
bool correlationProteins(cProtein *protein1, cProtein *protein2, const std::string &outputFilename);

bool getMatricesLinTransform(const std::string &first_slater_matrices,  const std::string &bessel_matrices,  const std::string &second_slater_matrices,  size_t order, double radius, double maxq, double alpha1, double beta1, double gamma1, double zshift, double alpha2, double beta2, double gamma2);
// Root mean square deviation
double rmsdQuality(const cProtein &protein, const cProtein &ethalon);
// Percent correct of chi_1
// (within 40 degrees those of the native)
double chi1Quality(const cProtein &protein, const cProtein &ethalon);
// Percent correct of chi_1 and chi_2
// (within 40 degrees those of the native)
double chi12Quality(const cProtein &protein, const cProtein &ethalon);
// Number of atom pairs in clash
size_t clashQuality(const cProtein &protein);
// Energy of the protein
double energyQuality(cProtein *protein);
// Some stuff
void doSmth(cProtein *protein);
