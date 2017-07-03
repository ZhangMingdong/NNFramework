#pragma once

// using which properties
enum EnumProperty
{
	P_User
	,
	P_Head
	,
	P_Expression
	,
	P_Eye
	,
	P_Photo
};
/*
// user
static EnumProperty g_targetProperty = P_User;
static char* g_pModelFileName = "user.net";
static int g_nHiddenLayers = 30;
static int g_nOutputLayers = 20;
*/


// head
static EnumProperty g_targetProperty = P_Head;
static char* g_pModelFileName = "head.net";
static int g_nHiddenLayers = 3;
static int g_nOutputLayers = 4;


/*
// expression
static EnumProperty g_targetProperty = P_Expression;
static char* g_pModelFileName = "expression.net";
static int g_nHiddenLayers = 8;
static int g_nOutputLayers = 4;
*/

/*
// eye
static EnumProperty g_targetProperty = P_Eye;
static char* g_pModelFileName = "sunglasses.net";
static int g_nHiddenLayers = 3;
static int g_nOutputLayers = 1;
*/


/*

static EnumProperty g_targetProperty = P_Eye;
static char* g_pModelFileName = "sunglasses.net";
static int g_nHiddenLayers = 3;
static int g_nOutputLayers = 1;

*/