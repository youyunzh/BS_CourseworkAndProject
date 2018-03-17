proj3: main.o bitmap.o
	g++ -o proj3 main.o bitmap.o
main.o: main.cpp
	g++ -std=c++11 -c main.cpp bynetwork.h
bitmap.o: bitmap.cpp
	g++ -std=c++11 -c bitmap.cpp
clean:
	rm proj3 main.o bitmap.o
