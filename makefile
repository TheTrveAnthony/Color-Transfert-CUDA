
# include openCV
cv = `pkg-config --cflags --libs opencv`

transfert : main.cpp function.cu
		nvcc $^ $(cv) -g -o $@ 