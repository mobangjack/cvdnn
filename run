if [ "$#" -eq "1" ]; then
   bash -exec "python cvdnn.py -i $1 -p bvlc_googlenet.prototxt -m bvlc_googlenet.caffemodel -l synset_words.txt"
else
   bash -exec "python cvdnn.py -p bvlc_googlenet.prototxt -m bvlc_googlenet.caffemodel -l synset_words.txt"
fi
