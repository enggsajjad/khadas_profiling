#!/bin/bash
inputfile="$1"
outputfile=${2:-'output_parse.txt'}

echo "Parsing $inputfile...."
if [ -z "$1" ]
  then
    {
    echo
    echo "ERROR: ABORTING \"parse.sh\", BECAUSE THE INPUT ARGUEMENT IS EMPTY!"
    echo "       MAYBE THE \"input parameter\" WAS NOT INCLUDED OR CONTAINS TYPING ERRORS."
    echo
    exit 1
  }
fi

echo -n "Create Neural Network: " > "$outputfile"
sed -n '/Create Neural Network/p' "$inputfile" | awk -F 'or | us' '{printf "%s",$2;}' >> "$outputfile"
echo >> "$outputfile"

echo -n "Verify Graph: " >> "$outputfile"
sed -n '/Verify Graph/p' "$inputfile" | awk -F 'or | us' '{printf "%s",$2;}' >> "$outputfile"
echo >> "$outputfile"

echo -n "Run the 1 time: " >> "$outputfile"
sed -n '/Run the 1 time/p' "$inputfile" | awk -F 'or | us' '{printf "%s",$2;}' >> "$outputfile"
echo >> "$outputfile"

echo -n "Total   : " >> "$outputfile"
sed -n '/Total   /p' "$inputfile" | awk -F 'or | us' '{printf "%s",$2;}' >> "$outputfile"
echo >> "$outputfile"

echo -n "Average : " >> "$outputfile"
sed -n '/Average /p' "$inputfile" | awk -F 'or | us' '{printf "%s",$2;}' >> "$outputfile"
echo >> "$outputfile"

sed -n '/dump layer :/,/us/{/dump layer :/d;p}' "$inputfile"  >> "$outputfile"


