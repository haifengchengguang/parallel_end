PROJ = test
CC = g++

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include
# LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
# LIBS = -lcppunit -ldl
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OMPFLAGS = -fopenmp
SIMDFLAGS = -O3 -lm

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cpp
	$(CC) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(SIMDFLAGS)

%.o : %.cpp %.h
	$(CC) $(OMPFLAGS)  $(CFLAGS) $< -o $@ $(SIMDFLAGS)

clean:
	rm -f $(PROJ) $(OBJS)