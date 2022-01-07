// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the zero parameter superpixel segmentation technique
// described in:
//
//
//
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua,
// and Sabine Susstrunk,
//
// IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//
//
//===========================================================================
// Copyright (c) 2013 Radhakrishna Achanta.
//
// For commercial use please contact the author:
//
// Email: firstname.lastname@epfl.ch
//===========================================================================

#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_

#include <vector>
#include <string>
#include <algorithm>
//向量化头文件
#include <immintrin.h>
#include <avx2intrin.h>
using namespace std;

class SLIC
{
public:
	SLIC();
	virtual ~SLIC();

	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void PerformSLICO_ForGivenK(
		const unsigned int *ubuff, // Each 32 bit unsigned int contains ARGB pixel values.
		const int width,
		const int height,
		int &numlabels,
		const int &K,
		const double &m,
		int &thread_count);

	//============================================================================
	// Save superpixel labels to pgm in raster scan order
	//============================================================================
	void SaveSuperpixelLabels2PPM(
		char *filename,
		const int width,
		const int height);

private:
	//============================================================================
	// Magic SLIC. No need to set M (compactness factor) and S (step size).
	// SLICO (SLIC Zero) varies only M dynamicaly, not S.
	//============================================================================
	void PerformSuperpixelSegmentation_VariableSandM(
		const int &STEP,
		const int &NUMITR);

	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		const int &STEP,
		const bool &perturbseeds,
		const vector<double> &edges);

	//============================================================================
	// Move the seeds to low gradient positions to avoid putting seeds at region boundaries.
	//============================================================================
	void PerturbSeeds(
		const vector<double> &edges);

	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const int &width,
		const int &height,
		vector<double> &edges);

	//============================================================================
	// xRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int &sR,
		const int &sG,
		const int &sB,
		double &X,
		double &Y,
		double &Z);

	//============================================================================
	// sRGB to CIELAB conversion
	//============================================================================
	void RGB2LAB(
		const int &sR,
		const int &sG,
		const int &sB,
		double &lval,
		double &aval,
		double &bval);

	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int *&ubuff);

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int &width,
		const int &height,
		int *nlabels,	// input labels that need to be corrected to remove stray labels
		int &numlabels, // the number of labels changes in the end if segments are removed
		const int &K);	// the number of superpixels desired by the user

private:
	int m_width;
	int m_height;
	int m_depth;

	double *m_lvec;
	double *m_avec;
	double *m_bvec;

	double **m_lvecvec;
	double **m_avecvec;
	double **m_bvecvec;
};
int thread_count;
#endif // !defined(_SLIC_H_INCLUDED_)
struct count_i
{
	double m_lvec;
	float m_avec;
	float m_bvec;
};
struct dists
{
	double distxy;
	float distlab;
	int klabels;
};
struct kseeds
{
	double l = 0;
	double a = 0;
	double b = 0;
	double x = 0;
	double y = 0;
};
count_i *param1;
vector<kseeds> param2(0);
dists *param3;

int fastMax(int x, int y) { return (((y - x) >> (32 - 1)) & (x ^ y)) ^ y; }
int fastMin(int x, int y) { return (((y - x) >> (32 - 1)) & (x ^ y)) ^ x; }

inline double fastPrecisePow(double a, double b) {
  // calculate approximation with fraction of the exponent
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  // exponentiation by squaring with the exponent's integer part
  // double r = u.d makes everything much slower, not sure why
  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }

  return r * u.d;
}
inline double ga(double x){
	return (x + 0.055) / 1.055;
}
//泰勒逼近
double mypow(double a, double b)
{
	bool gt1 = (sqrt((a-1)*(a-1)) <= 1)? false:true;

	int oc = -1; // used to alternate math symbol (+,-)
	int iter = 20; // number of iterations
	double p, x, x2, sumX, sumY;

	if( (b-floor(b)) == 0 )
	{
// return base^exponent
		p = a;
		for( int i = 1; i < b; i++ )p *= a;
		return p;
	}
	x = (gt1)?
		(a /(a-1)): // base is greater than 1
			(a-1); // base is 1 or less
	sumX = (gt1)?
			(1/x): // base is greater than 1
			x; // base is 1 or less
	for( int i = 2; i < iter; i++ )
	{
// find x^iteration
		p = x;
		for( int j = 1; j < i; j++)p *= x;
		double xTemp = (gt1)?
				(1/(i*p)): // base is greater than 1
				(p/i); // base is 1 or less
		sumX = (gt1)?
				(sumX+xTemp): // base is greater than 1
				(sumX+(xTemp*oc)); // base is 1 or less
		oc *= -1; // change math symbol (+,-)
	}
	x2 = b * sumX;
	sumY = 1+x2; // our estimate
	for( int i = 2; i <= iter; i++ )
	{
		// find x2^iteration
		p = x2;
		for( int j = 1; j < i; j++)p *= x2;
		// multiply iterations (ex: 3 iterations = 3*2*1)
		int yTemp = 2;
		for( int j = i; j > 2; j-- )yTemp *= j;
		// add to estimate (ex: 3rd iteration => (x2^3)/(3*2*1) )
		sumY += p/yTemp;
	}
	return sumY; // return our estimate
}



#define TASKS 1024

#ifndef ARCH_HAS_PREFETCH
#define prefetch(x) __builtin_prefetch(x)
#endif
 
static inline void prefetch_range(void *addr, size_t len)
{
#ifdef ARCH_HAS_PREFETCH
    char *cp;
    char *end = addr + len;
 
    for (cp = addr; cp < end; cp += PREFETCH_STRIDE)
        prefetch(cp);
#endif
}