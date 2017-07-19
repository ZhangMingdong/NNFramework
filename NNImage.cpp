#include "NNImage.h"
#include <qfile.h>
#include "setting.h"
#include <assert.h>

NNImage::NNImage()
{
}


void imgl_munge_name(char *buf)
{
	int j;

	j = 0;
	while (buf[j] != '\n') j++;
	buf[j] = '\0';
}

char *img_basename(char *filename)
{
	int len = strlen(filename);
	/*
	char* buf = new char[len];
	strcpy(buf, filename);
	return buf;
	*/
	int dex = len - 1;
	while (dex > -1) {
		if (filename[dex] == '/' || filename[dex] == '\\') {
			break;
		}
		else {
			dex--;
		}
	}
	dex++;
	char* part = &(filename[dex]);
	len = strlen(part);
	char* newptr = new char[len + 1];
	strcpy(newptr, part);
	int n = strcmp(filename, newptr);

	return(newptr);
}

NNImage::~NNImage()
{
	if (_arrData) delete[] _arrData;
	if (_arrName) delete[] _arrName;
}

void NNImage::SetPixel(int r, int c,int t, int val)
{
	int nc = this->_nCols;
	int nt = this->_nTuls;
	this->_arrData[((r * nc) + c)*nt + t] = val;
}

int NNImage::GetPixel(int r, int c,int t)
{
	int nc = this->_nCols;
	int nt = this->_nTuls;
	return (this->_arrData[((r * nc) + c)*nt + t]);
}

void NNImage::setLabelFromName() {
	int scale;
	char userid[40], head[40], expression[40], eyes[40], photo[40];

	userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

	/*** scan in the image features ***/
	sscanf(_arrName, "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]",
		userid, head, expression, eyes, &scale, photo);

	char *p = strrchr(eyes, '.');
	if (p != NULL)
	{
		userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

		sscanf(_arrName, "%[^_]_%[^_]_%[^_]_%[^.].%[^_]", userid, head, expression, eyes, photo);
		scale = 1;
	}

	p = strrchr(userid, '\\');
	if (p != NULL)
		p++;

	switch (g_targetProperty)
	{
	case P_User:
		_nK = 20;
		if (!strcmp(p, "an2i")) {
			_nLabel = 1;
		}
		else if (!strcmp(p, "at33")) {
			_nLabel = 2;
		}
		else if (!strcmp(p, "boland")) {
			_nLabel = 3;
		}
		else if (!strcmp(p, "bpm")) {
			_nLabel = 4;
		}
		else if (!strcmp(p, "ch4f")) {
			_nLabel = 5;
		}
		else if (!strcmp(p, "cheyer")) {
			_nLabel = 6;
		}
		else if (!strcmp(p, "choon")) {
			_nLabel = 7;
		}
		else if (!strcmp(p, "danieln")) {
			_nLabel = 8;
		}
		else if (!strcmp(p, "glickman")) {
			_nLabel = 9;
		}
		else if (!strcmp(p, "karyadi")) {
			_nLabel = 10;
		}
		else if (!strcmp(p, "kawamura")) {
			_nLabel = 11;
		}
		else if (!strcmp(p, "kk49")) {
			_nLabel = 12;
		}
		else if (!strcmp(p, "megak")) {
			_nLabel = 13;
		}
		else if (!strcmp(p, "mitchell")) {
			_nLabel = 14;
		}
		else if (!strcmp(p, "night")) {
			_nLabel = 15;
		}
		else if (!strcmp(p, "phoebe")) {
			_nLabel = 16;
		}
		else if (!strcmp(p, "saavik")) {
			_nLabel = 17;
		}
		else if (!strcmp(p, "steffi")) {
			_nLabel = 18;
		}
		else if (!strcmp(p, "sz24")) {
			_nLabel = 19;
		}
		else if (!strcmp(p, "tammo")) {
			_nLabel = 20;
		}
		else {
			assert(false);
			exit(0);
		}
		break;
	case P_Head:
		_nK = 4;
		if (!strcmp(head, "up")) {
			_nLabel = 1;
		}
		else if (!strcmp(expression, "left")) {
			_nLabel = 2;
		}
		else if (!strcmp(expression, "straight")) {
			_nLabel = 3;
		}
		else {
			_nLabel = 4;
		}
		break;
	case P_Expression:
		_nK = 4;
		if (!strcmp(expression, "angry")) {
			_nLabel = 1;
		}
		else if (!strcmp(expression, "happy")) {
			_nLabel = 2;
		}
		else if (!strcmp(expression, "neutral")) {
			_nLabel = 3;
		}
		else {
			_nLabel = 4;
		}
		break;
	case P_Eye:
		_nK = 2;
		if (!strcmp(eyes, "sunglasses")) {
			_nLabel = 1;
		}
		else {
			_nLabel = 2;
		}
		break;

	default:
		break;
	}
	
}

NNImage* NNImage::Read(char *filename)
{
	FILE *pgm = fopen(filename, "r");

	if (pgm == NULL) {
		printf("IMGOPEN: Couldn't open '%s'\n", filename);
		return(NULL);
	}


	NNImage *newptr = new NNImage();
	char line[512], intbuf[100], ch;
	int type, nc, nr, maxval, i, j, k, found;

	newptr->_arrName = img_basename(filename);

	/*** Scan pnm type information, expecting P5 ***/
	fgets(line, 511, pgm);
	sscanf(line, "P%d", &type);
	if (type != 5 && type != 2) {
		printf("IMGOPEN: Only handles pgm files (type P5 or P2)\n");
		fclose(pgm);
		return(NULL);
	}

	/*** Get dimensions of pgm ***/
	fgets(line, 511, pgm);
	sscanf(line, "%d %d", &nc, &nr);
	newptr->_nRows = nr;
	newptr->_nCols = nc;

	/*** Get maxval ***/
	fgets(line, 511, pgm);
	sscanf(line, "%d", &maxval);
	if (maxval > 255) {
		printf("IMGOPEN: Only handles pgm files of 8 bits or less\n");
		fclose(pgm);
		return(NULL);
	}

	newptr->_arrData = new int[newptr->_nTuls*nr * nc];
	if (newptr->_arrData == NULL) {
		printf("IMGOPEN: Couldn't allocate space for image data\n");
		fclose(pgm);
		return(NULL);
	}

	if (type == 5) {

		for (i = 0; i < nr; i++) {
			for (j = 0; j < nc; j++) {
				newptr->SetPixel(i, j,0, fgetc(pgm));
			}
		}

	}
	else if (type == 2) {

		for (i = 0; i < nr; i++) {
			for (j = 0; j < nc; j++) {

				k = 0;  found = 0;
				while (!found) {
					ch = (char)fgetc(pgm);
					if (ch >= '0' && ch <= '9') {
						intbuf[k] = ch;  k++;
					}
					else {
						if (k != 0) {
							intbuf[k] = '\0';
							found = 1;
						}
					}
				}
				newptr->SetPixel(i, j,0, atoi(intbuf));

			}
		}

	}
	else {
		printf("IMGOPEN: Fatal impossible error\n");
		fclose(pgm);
		return (NULL);
	}

	fclose(pgm);
	newptr->setLabelFromName();
	return (newptr);
}

void NNImageList::LoadFromFile(char *filename)
{
	char buf[2000];

	if (filename[0] == '\0') {
		printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Invalid file '%s'\n", filename);
		return;
	}
	FILE *fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("IMGL_LOAD_IMAGES_FROM_TEXTFILE: Couldn't open '%s'\n", filename);
		return;
	}

	while (fgets(buf, 1999, fp) != NULL) {
		imgl_munge_name(buf);
		printf("Loading '%s'...", buf); 
		NNImage *iimg = NNImage::Read(buf);
		if (iimg == 0) {
			printf("Couldn't open '%s'\n", buf);
		}
		else {
			this->push_back(iimg);
			printf("done\n");
		}
	}

	fclose(fp);
}


void NNImageList::LoadFromFileNew(const char *filename) {
	// length of the data is 30730000
	// which is (1+1024*3)*10000
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly)) {
		return;
	}

	char* temp = new char[file.size()];
	file.read(temp, file.size());

	unsigned char* arrData = (unsigned char*)temp;
	int x = file.size();
	for (size_t i = 0; i < g_nImgs; i++)
	{
		NNImage *iimg = new NNImage();
		iimg->_nLabel = *arrData++;
		iimg->_nRows = g_nRow;
		iimg->_nCols = g_nCol;
		iimg->_nTuls = 3;
		iimg->_arrData = new int[iimg->_nRows*iimg->_nCols*iimg->_nTuls];
		for (size_t k = 0; k < 3; k++)
		{
			for (int i = 0; i < g_nRow; i++)
			{
				for (size_t j = 0; j < g_nCol; j++)
				{
					iimg->SetPixel(i, j, k, *arrData++);
				}
			}
		}
		this->push_back(iimg);
	}
}