CXX = clang++
CFLAGS = -O2 -g -fPIC -std=c++11
INCL = -I../include `python-config --includes`

akima: PreCompPy.o
	$(CXX) $(INCL) $(CFLAGS) -dynamiclib -undefined dynamic_lookup -shared PreCompPy.o -o _precomp.so

.cpp.o:
	$(CXX) $(INCL) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o _precomp.so
