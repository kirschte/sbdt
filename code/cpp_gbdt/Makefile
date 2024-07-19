COMPILER=g++
CFLAGS=-c -Wall -Wextra -std=gnu++11 -g3 -O0 -pipe -D_GLIBCXX_DEBUG
MAKEFLAGS+="-j 12"
SOURCES=$(wildcard src/*.cpp src/gbdt/*.cpp)
INC=-I./include/ -I./include/gbdt/
LINKFLAG=-Wl,--no-as-needed -pthread -lfmt
MAKROFLAG=#-DDELTA=5e-8 -DMAX_ALPHA=1000

$(shell mkdir -p objs)
$(shell mkdir -p objs/gbdt)
.PHONY: clean rebuild f fast fs fast_shared

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) $(LINKFLAG) $^ -o $@ 

libsbdt.a: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) $(LINKFLAG) $^ -o $@

clean:
	$(RM) -r *~ objs run *.a

objs/%.o: src/%.cpp
	$(COMPILER) $(CFLAGS) $(MAKROFLAG) $(INC) $< -o $@
objs/gbdt/%.o: src/gbdt/%.cpp
	$(COMPILER) $(CFLAGS) $(MAKROFLAG) $(INC) $< -o $@

rebuild:
	make

# let the compiler do it's magic
f fast:
	make CFLAGS="-c -Werror -std=gnu++11 -Ofast -pipe -ffast-math -march=native -mveclibabi=svml -pthread -flto"

fs fast_shared:
	make libsbdt.a CFLAGS="-c -Werror -std=gnu++11 -Ofast -pipe -ffast-math -march=native -mveclibabi=svml -pthread -flto -fPIC" LINKFLAG="$(LINKFLAG) -lsvml -shared"
