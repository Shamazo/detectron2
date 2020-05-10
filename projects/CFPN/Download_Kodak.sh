for i in {01..24..1}; do
  echo ${i}
  wget http://r0k.us/graphics/kodak/kodak/kodim${i}.png -O datasets/kodak/${i}.png
done

for i in {1..9}; do
  echo ${i}
  mv datasets/kodak/0${i}.png datasets/kodak/${i}.png
done