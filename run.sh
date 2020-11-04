FILE=$PWD/src

if [ -d "$FILE" ]; then
    export PYTHONPATH=$FILE:$PYTHONPATH
else
    echo "[ERROR] $FILE is nonexistent"
fi

python3 $1