echo "Checking Kaggle Install..."
if ! command -v kaggle &> /dev/null
then
	echo "Kaggle is not found"
	pip3 install --user kaggle
else
	echo "Kaggle is installed"
fi

echo "Checking Kingbase Dataset... (if this fails, check that GSUtil is installed and authenticated)"
if [ -e "./data/datasets/kingbase-ftfy.txt" ]
then
	echo "Kingbase Dataset Exists"
else
	gsutil cp gs://gpt-2-poetry/data/kingbase-ftfy.txt ./data/datasets/kingbase-ftfy.txt
fi

echo "Checking Kaggle Dataset..."

if [ -e "./data/datasets/35-million-chess-games.zip" ]
then
	echo "Kaggle Dataset Downloaded"
else
	cd data/datasets 
	~/.local/bin/kaggle datasets download milesh1/35-million-chess-games
	cd ../..
fi

if [ -e "./data/datasets/all_with_filtered_anotations_since1998.txt" ]
then
	echo "Kaggle Dataset Unzipped"
else
	unzip ./data/datasets/35-million-chess-games.zip -d ./data/datasets/
fi
