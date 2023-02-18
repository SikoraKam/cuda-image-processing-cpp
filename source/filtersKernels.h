#pragma once

// blur
int radius1 = 1;
int weight1 = 9;

 /*
 @brief represents kernel for blur. Values should be divied by 9, so weight should equal 9
 */
int kernel1[] =
{
  1, 1, 1,
  1, 1 ,1,
  1, 1, 1
};


int radius2 = 2;
int weight2 = 273;
/*
@brief represents kernel for gaussian blur
*/
int kernel2[] =
{
    1, 4, 7, 4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1, 4, 7, 4, 1
};



int radius3 = 1;
int weight3 = 1;
/*
@brief represents kernel for detecting horizontal edges
*/
int kernel3[] =
{
   1, 2, 1,
   0, 0 ,0,
    -1, -2, -1
};

int radius4 = 1;
int weight4 = 1;
/*
@brief represents kernel for detecting vertical edges
*/
int kernel4[] =
{
  -1, 0, 1,
  -2, 0, 2
  -1, 0, 1
};

int radius5 = 1;
int weight5 = 1;
/*
@brief represents kernel for detecting edges
*/
int kernel5[] =
{
  -1, -1, -1,
  -1,  8, -1,
  -1, -1, -1
};


int radius6 = 1;
int weight6 = 1;
/*
@brief represents kernel sharp conversion. Some of the edges should be more visible
*/
int kernel6[] =
{
  -1, -1, -1,
  -1,  9, -1,
  -1, -1, -1
};



