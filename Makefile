all:
	gcc -Wall -g `pkg-config --cflags opencv` `pkg-config --libs opencv` -o app main.cpp  

