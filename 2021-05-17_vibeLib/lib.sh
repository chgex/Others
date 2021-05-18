

g++ -o vibe.so -shared -fPIC *.cpp `pkg-config --cflags --libs opencv`
echo >> "vibe.so "

# cd ..
# python use.py

