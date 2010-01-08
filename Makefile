all:
	gcc -Wall `pkg-config --cflags opencv` `pkg-config --libs opencv` -o app main.cpp  

