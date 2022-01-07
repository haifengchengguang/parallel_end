#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "finalTest.h"
#include <chrono>
#include <omp.h>
typedef chrono::high_resolution_clock Clock;

// For superpixels
const int dx4[4] = {-1, 0, 1, 0};
const int dy4[4] = {0, -1, 0, 1};
// const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
// const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1, 0, 1, 0, -1, 1, 1, -1, 0, 0};
const int dy10[10] = {0, -1, 0, 1, -1, -1, 1, 1, 0, 0};
const int dz10[10] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 1};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SLIC::SLIC()
{
    m_lvec = NULL;
    m_avec = NULL;
    m_bvec = NULL;

    m_lvecvec = NULL;
    m_avecvec = NULL;
    m_bvecvec = NULL;
}

SLIC::~SLIC()
{
    if (m_lvec)
        delete[] m_lvec;
    if (m_avec)
        delete[] m_avec;
    if (m_bvec)
        delete[] m_bvec;

    if (m_lvecvec)
    {
        for (int d = 0; d < m_depth; d++)
            delete[] m_lvecvec[d];
        delete[] m_lvecvec;
    }
    if (m_avecvec)
    {
        for (int d = 0; d < m_depth; d++)
            delete[] m_avecvec[d];
        delete[] m_avecvec;
    }
    if (m_bvecvec)
    {
        for (int d = 0; d < m_depth; d++)
            delete[] m_bvecvec[d];
        delete[] m_bvecvec;
    }
}

void SLIC::RGB2XYZ(
    const int &sR,
    const int &sG,
    const int &sB,
    double &X,
    double &Y,
    double &Z)
{
    double R = sR / 255.0;
    double G = sG / 255.0;
    double B = sB / 255.0;

    double r, g, b;

    if (R <= 0.04045)
        r = R / 12.92;
    else
        r = fastPrecisePow((R + 0.055) / 1.055, 2.4);
    if (G <= 0.04045)
        g = G / 12.92;
    else
        g = fastPrecisePow((G + 0.055) / 1.055, 2.4);
    if (B <= 0.04045)
        b = B / 12.92;
    else
        b = fastPrecisePow((B + 0.055) / 1.055, 2.4);

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

void SLIC::DoRGBtoLABConversion(
    const unsigned int *&ubuff)
{
    int sz = m_width * m_height;

#pragma omp parallel for num_threads(thread_count)
    for (int j = 0; j < sz; j++)
    {
        int r = (ubuff[j] >> 16) & 0xFF;
        int g = (ubuff[j] >> 8) & 0xFF;
        int b = (ubuff[j]) & 0xFF;

        // RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
        double X, Y, Z;
        // RGB2XYZ(sR, sG, sB, X, Y, Z);
        double R = r / 255.0;
        double G = g / 255.0;
        double B = b / 255.0;

        double r1, g1, b1;

        if (R <= 0.04045)
            r1 = R / 12.92;
        else
            r1 = pow((R + 0.055) / 1.055, 2.4);
        if (G <= 0.04045)
            g1 = G / 12.92;
        else
            g1 = pow((G + 0.055) / 1.055, 2.4);
        if (B <= 0.04045)
            b1 = B / 12.92;
        else
            b1 = pow((B + 0.055) / 1.055, 2.4);

        X = r1 * 0.4124564 + g1 * 0.3575761 + b1 * 0.1804375;
        Y = r1 * 0.2126729 + g1 * 0.7151522 + b1 * 0.0721750;
        Z = r1 * 0.0193339 + g1 * 0.1191920 + b1 * 0.9503041;
        //------------------------
        // XYZ to LAB conversion
        //------------------------
        double epsilon = 0.008856; // actual CIE standard
        double kappa = 903.3;      // actual CIE standard

        double Xr = 0.950456; // reference white
        double Yr = 1.0;      // reference white
        double Zr = 1.088754; // reference white

        double xr = X / Xr;
        double yr = Y / Yr;
        double zr = Z / Zr;

        double fx, fy, fz;
        double div = 1/116.0;
        if (xr > epsilon)
            fx = cbrt(xr); // pow改为cbrt
        else
            fx = (kappa * xr + 16.0) *div;
        if (yr > epsilon)
            fy = cbrt(yr);
        else
            fy = (kappa * yr + 16.0) *div;
        if (zr > epsilon)
            fz = cbrt(zr);
        else
            fz = (kappa * zr + 16.0) *div;

        param1[j].m_lvec = 116.0 * fy - 16.0;
        param1[j].m_avec = 500.0 * (fx - fy);
        param1[j].m_bvec = 200.0 * (fy - fz);
    }

}

void SLIC::PerformSLICO_ForGivenK(
    const unsigned int *ubuff,
    const int width,
    const int height,
    int &numlabels,
    const int &K, // required number of superpixels
    const double &m,
    int &thread_count) // weight given to spatial distance
{

    //--------------------------------------------------
    m_width = width;
    m_height = height;
    int sz = m_width * m_height;
    //--------------------------------------------------
    // if(0 == klabels) klabels = new int[sz];

#pragma omp parallel for num_threads(thread_count)
    for (int s = 0; s < sz; s++)
        param3[s].klabels = -1;

    //--------------------------------------------------
    if (1) // LAB
    {
    
        DoRGBtoLABConversion(ubuff);
     
    }
    //--------------------------------------------------

    bool perturbseeds(true);
    vector<double> edgemag(0);

    if (perturbseeds)
        DetectLabEdges(m_width, m_height, edgemag);

   
    GetLABXYSeeds_ForGivenK(K, perturbseeds, edgemag);

    int STEP = sqrt(double(sz) / double(K)) + 2.0; // adding a small value in the even the STEP size is too small.


    PerformSuperpixelSegmentation_VariableSandM(STEP, 10);

    numlabels = param2.size();

    int *nlabels = new int[sz];

    EnforceLabelConnectivity(m_width, m_height, nlabels, numlabels, K);

#pragma omp parallel for
    for (int i = 0; i < sz; i++)
        param3[i].klabels = nlabels[i];
    if (nlabels)
        delete[] nlabels;
}

void SLIC::GetLABXYSeeds_ForGivenK(
    const int &K,
    const bool &perturbseeds,
    const vector<double> &edgemag)
{
    int sz = m_width * m_height;
    double step = sqrt(double(sz) / double(K));
    int T = step;
    int xoff = step / 2;
    int yoff = step / 2;

    int n(0);
    int r(0);
    for (int y = 0; y < m_height; y++)
    {
        int Y = y * step + yoff;
        if (Y > m_height - 1)
            break;

        for (int x = 0; x < m_width; x++)
        {
            // int X = x*step + xoff;//square grid
            int X = x * step + (xoff << (r & 0x1)); // hex grid
            if (X > m_width - 1)
                break;

            int i = Y * m_width + X;

            kseeds temp;
            temp.l = param1[i].m_lvec;
            temp.a = param1[i].m_avec;
            temp.b = param1[i].m_bvec;
            temp.x = X;
            temp.y = Y;
            param2.push_back(temp);
            n++;
        }
        r++;
    }

    if (perturbseeds)
    {
        PerturbSeeds(edgemag);
    }
}
void SLIC::PerturbSeeds(
    const vector<double> &edges)
{
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    int numseeds = param2.size();

    for (int n = 0; n < numseeds; n++)
    {
        int ox = param2[n].x; // original x
        int oy = param2[n].y; // original y
        int oind = oy * m_width + ox;

        int storeind = oind;
        for (int i = 0; i < 8; i++)
        {
            int nx = ox + dx8[i]; // new x
            int ny = oy + dy8[i]; // new y

            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
            {
                int nind = ny * m_width + nx;
                if (edges[nind] < edges[storeind])
                {
                    storeind = nind;
                }
            }
        }
        if (storeind != oind)
        {
            param2[n].x = storeind % m_width;
            param2[n].y = storeind / m_width;
            param2[n].l = param1[storeind].m_lvec;
            param2[n].a = param1[storeind].m_avec;
            param2[n].b = param1[storeind].m_bvec;
        }
    }
}

void SLIC::PerformSuperpixelSegmentation_VariableSandM(
    const int &STEP,
    const int &NUMITR)
{
    int sz = m_width * m_height;
    const int numk = param2.size();
    // double cumerr(99999.9);
    int numitr(0);

    //----------------
    int offset = STEP;
    if (STEP < 10)
        offset = STEP * 1.5;
    //----------------

    vector<double> sigmal(numk, 0);
    vector<double> sigmaa(numk, 0);
    vector<double> sigmab(numk, 0);
    vector<double> sigmax(numk, 0);
    vector<double> sigmay(numk, 0);
    vector<int> clustersize(numk, 0);
    vector<double> inv(numk, 0); // to store 1/clustersize[k] values
    vector<double> distvec(sz, DBL_MAX);
    vector<float> maxlab(numk, 10 * 10);    // THIS IS THE VARIABLE VALUE OF M, just start with 10
    vector<float> maxxy(numk, STEP * STEP); // THIS IS THE VARIABLE VALUE OF M, just start with 10

    double invxywt = 1.0 / (STEP * STEP); // NOTE: this is different from how usual SLIC/LKM works
    while (numitr < NUMITR)
    {
        //------
        // cumerr = 0;
        numitr++;
        //------

        distvec.assign(sz, DBL_MAX);
    
        int task_count = ceil((double)sz/(double)TASKS);
        #pragma omp parallel num_threads(thread_count)
        {   
            int N = omp_get_num_threads();
            int id = omp_get_thread_num();
            int dis = task_count / N;
            int start = id * dis;
            int end;

            if (id == N - 1)
            {
                end = task_count;
            }
            else
            {
                end = start + dis;
            }

            for(int l=start;l<end;l++)
            {
                
                int s_start,s_end;
                s_start = l*TASKS;
                if(l == task_count){
                    s_end = sz;
                }else{
                    s_end = s_start+TASKS;
                }
                int start_y = s_start/m_width;
                int numk_dis = numk/10;
                int mod = numk%10;
                    for(int m=0;m<10;m++){
                        int tem = numk_dis*m;
                        int tem1 = tem+numk_dis;
                        if(m==9) tem1 +=mod;
                        //prefetch_range((void *)&param2[tem],sizeof(param2)*numk_dis);
                        for(int n=tem;n<tem1;n++){
                            int y1 = fastMax(start_y, (int)(param2[n].y - offset));
                    int x1 = fastMax(0, (int)(param2[n].x - offset));
                    int y2 = fastMin(m_height, (int)(param2[n].y + offset));
                    int x2 = fastMin(m_width, (int)(param2[n].x + offset));

                    int left = y2*m_width+x2;
                    if(s_start > left) continue;
                    int right = y1*m_width+x1;
                    if(s_end <= right) continue;

                    for (int y = y1; y < y2; y++)
                    {
                        for (int x = x1; x < x2; x++)
                        {
                            int i = y * m_width + x;
                            if (i < s_start)
                            {
                                continue;
                            }
                            if(i >= s_end){
                                y = y2;
                                break;
                            }
                            double my_xy = (x - param2[n].x) * (x - param2[n].x);
                            my_xy += (y - param2[n].y) * (y - param2[n].y);
                            param3[i].distxy = my_xy;

                            double l = param1[i].m_lvec;
                            double a = param1[i].m_avec;
                            double b = param1[i].m_bvec;

                            param3[i].distlab = (l - param2[n].l) * (l - param2[n].l) +
                                                (a - param2[n].a) * (a - param2[n].a) +
                                                (b - param2[n].b) * (b - param2[n].b);
    
                            double dist = param3[i].distlab * (1/ maxlab[n]) + param3[i].distxy * invxywt; 
                        
                            if (dist < distvec[i])
                            {
                                distvec[i] = dist;
                                param3[i].klabels = n;
                            }
                        }
                    }
                        }
                    }
            }
        }
     
        for (int i = 0; i < sz; i++)
        {
            if (maxlab[param3[i].klabels] < param3[i].distlab)
                maxlab[param3[i].klabels] = param3[i].distlab;
            if (maxxy[param3[i].klabels] < param3[i].distxy)
                maxxy[param3[i].klabels] = param3[i].distxy;
        }
    
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        sigmal.assign(numk, 0);
        sigmaa.assign(numk, 0);
        sigmab.assign(numk, 0);
        sigmax.assign(numk, 0);
        sigmay.assign(numk, 0);
        clustersize.assign(numk, 0);
       
  
        for (int j = 0; j < sz; j++)
        {
            int temp = param3[j].klabels;
            //_ASSERT(klabels[j] >= 0);
            sigmal[temp] += param1[j].m_lvec;
            sigmaa[temp] += param1[j].m_avec;
            sigmab[temp] += param1[j].m_bvec;
            sigmax[temp] += (j % m_width);
            sigmay[temp] += (j / m_width);

            clustersize[temp]++;
        }
       
      
        {
            for (int k = 0; k < numk; k++)
            {
                //_ASSERT(clustersize[k] > 0);
                if (clustersize[k] <= 0)
                    clustersize[k] = 1;
                inv[k] = 1.0 / double(clustersize[k]); // computing inverse now to multiply, than divide later
            }
        }
     
        {
            for (int k = 0; k < numk; k++) //这里可以稍微向量化
            {
                param2[k].l = sigmal[k] * inv[k];
                param2[k].a = sigmaa[k] * inv[k];
                param2[k].b = sigmab[k] * inv[k];
                param2[k].x = sigmax[k] * inv[k];
                param2[k].y = sigmay[k] * inv[k];
            }
        }
        
    }
}

void SLIC::EnforceLabelConnectivity(
    const int &width,
    const int &height,
    int *nlabels,   // new labels
    int &numlabels, // the number of labels changes in the end if segments are removed
    const int &K)   // the number of superpixels desired by the user
{
    //	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    //	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    const int sz = width * height;
    const int SUPSZ = sz / K;
    // nlabels.resize(sz, -1);
    for (int i = 0; i < sz; i++)
        nlabels[i] = -1;
    int label(0);
    int *xvec = new int[sz];
    int *yvec = new int[sz];
    int oindex(0);
    int adjlabel(0); // adjacent label

    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            if (0 > nlabels[oindex])
            {
                nlabels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            int nindex = y * width + x;
                            if (nlabels[nindex] >= 0)
                                adjlabel = nlabels[nindex];
                        }
                    }
                }

                int count(1);
                for (int c = 0; c < count; c++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            int nindex = y * width + x;

                            if (0 > nlabels[nindex] && param3[oindex].klabels == param3[nindex].klabels)
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if (count <= SUPSZ >> 2)
                {
                    for (int c = 0; c < count; c++)
                    {
                        int ind = yvec[c] * width + xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    numlabels = label;

    if (xvec)
        delete[] xvec;
    if (yvec)
        delete[] yvec;
}

void SLIC::SaveSuperpixelLabels2PPM(
    char *filename,
    const int width,
    const int height)
{
    FILE *fp;
    char header[20];

    fp = fopen(filename, "wb");

    // write the PPM header info, such as type, width, height and maximum
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // write the RGB data
    unsigned char *rgb = new unsigned char[(width) * (height)*3];
    int k = 0;
    unsigned char c = 0;
    for (int i = 0; i < (height); i++)
    {
        for (int j = 0; j < (width); j++)
        {
            c = (unsigned char)(param3[k].klabels);
            rgb[i * (width)*3 + j * 3 + 2] = param3[k].klabels >> 16 & 0xff; // r
            rgb[i * (width)*3 + j * 3 + 1] = param3[k].klabels >> 8 & 0xff;  // g
            rgb[i * (width)*3 + j * 3 + 0] = param3[k].klabels & 0xff;       // b

            // rgb[i*(width) + j + 0] = c;
            k++;
        }
    }
    fwrite(rgb, width * height * 3, 1, fp);

    delete[] rgb;

    fclose(fp);
}

void LoadPPM(char *filename, unsigned int **data, int *width, int *height)
{
    char header[1024];
    FILE *fp = NULL;
    int line = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2)
    {
        fgets(header, 1024, fp);
        if (header[0] != '#')
        {
            ++line;
        }
    }
    // read width and height
    sscanf(header, "%d %d\n", width, height);

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char *rgb = new unsigned char[(*width) * (*height) * 3];
    fread(rgb, (*width) * (*height) * 3, 1, fp);

    *data = new unsigned int[(*width) * (*height) * 4];
    int k = 0;
    for (int i = 0; i < (*height); i++)
    {
        for (int j = 0; j < (*width); j++)
        {
            unsigned char *p = rgb + i * (*width) * 3 + j * 3;
            // a ( skipped )
            (*data)[k] = p[2] << 16; // r
            (*data)[k] |= p[1] << 8; // g
            (*data)[k] |= p[0];      // b
            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete[] rgb;

    fclose(fp);
}

int CheckLabelswithPPM(char *filename, int width, int height)
{
    char header[1024];
    FILE *fp = NULL;
    int line = 0, ground = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2)
    {
        fgets(header, 1024, fp);
        if (header[0] != '#')
        {
            ++line;
        }
    }
    // read width and height
    int w(0);
    int h(0);
    sscanf(header, "%d %d\n", &w, &h);
    if (w != width || h != height)
        return -1;

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char *rgb = new unsigned char[(w) * (h)*3];
    fread(rgb, (w) * (h)*3, 1, fp);

    int num = 0, k = 0;
    for (int i = 0; i < (h); i++)
    {
        for (int j = 0; j < (w); j++)
        {
            unsigned char *p = rgb + i * (w)*3 + j * 3;
            // a ( skipped )
            ground = p[2] << 16; // r
            ground |= p[1] << 8; // g
            ground |= p[0];      // b

            if (ground != param3[k].klabels)
                num++;

            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete[] rgb;

    fclose(fp);

    return num;
}

void SLIC::DetectLabEdges(
    const int &width,
    const int &height,
    vector<double> &edges)
{
    int sz = width * height;

    edges.resize(sz, 0);
#pragma omp parallel for collapse(2) num_threads(thread_count)
    for (int j = 1; j < height - 1; j++)
    {
        for (int k = 1; k < width - 1; k++)
        {
            int i = j * width + k;

            double dx = (param1[i - 1].m_lvec - param1[i + 1].m_lvec) * (param1[i - 1].m_lvec - param1[i + 1].m_lvec) +
                        (param1[i - 1].m_avec - param1[i + 1].m_avec) * (param1[i - 1].m_avec - param1[i + 1].m_avec) +
                        (param1[i - 1].m_bvec - param1[i + 1].m_bvec) * (param1[i - 1].m_bvec - param1[i + 1].m_bvec);

            double dy = (param1[i - width].m_lvec - param1[i + width].m_lvec) * (param1[i - width].m_lvec - param1[i + width].m_lvec) +
                        (param1[i - width].m_avec - param1[i + width].m_avec) * (param1[i - width].m_avec - param1[i + width].m_avec) +
                        (param1[i - width].m_bvec - param1[i + width].m_bvec) * (param1[i - width].m_bvec - param1[i + width].m_bvec);

            // edges[i] = (sqrt(dx) + sqrt(dy));
            edges[i] = (dx + dy);
        }
    }
}

int main(int argc, char **argv)
{
    unsigned int *img = NULL;
    int width(0);
    int height(0);
    LoadPPM((char *)"input_image.ppm", &img, &width, &height);
    if (width == 0 || height == 0)
        return -1;
    int sz = width * height;
    param1 = new count_i[sz];
    param3 = new dists[sz];
    int numlabels(0);
    SLIC slic;
    int m_spcount;
    double m_compactness;
    m_spcount = 200;
    m_compactness = 10.0;
    thread_count = strtol(argv[1], NULL, 10);
    auto startTime = Clock::now();
    slic.PerformSLICO_ForGivenK(img, width, height, numlabels, m_spcount, m_compactness, thread_count); // for a given number K of superpixels
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() / 1000 << " ms" << endl;

    int num = CheckLabelswithPPM((char *)"check.ppm", width, height);
    if (num < 0)
    {
        cout << "The result for labels is different from output_labels.ppm." << endl;
    }
    else
    {
        cout << "There are " << num << " points' labels are different from original file." << endl;
    }

    slic.SaveSuperpixelLabels2PPM((char *)"output_labels.ppm", width, height);
    if (param1)
        delete[] param1;

    if (img)
        delete[] img;

    return 0;
}