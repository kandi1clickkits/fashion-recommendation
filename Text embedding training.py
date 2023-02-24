from sentence_transformers import SentenceTransformer
import pickle
import csv

INP_FILE_PATH="styles.csv"
n_rows = 20_000

header =True
list_prod_id = list(); list_prod_dispname = list(); list_prod_disp_name = list()
with open(INP_FILE_PATH) as file:
    rows = csv.reader(file)
    for row in rows:
        if header: header = False; continue
        else:
            list_prod_id.append(row[0])
            list_prod_disp_name.append(row[9])
            if row[2]=="Apparel":
                list_prod_dispname.append(row[6]+" "+row[8]+" "+ row[9]+  " and "+  row[2] + " for " +row[1])
            else:
                list_prod_dispname.append(row[9]+ " "+ row[2] + " and "+  row[3] +" for " +row[1])
list_prod_dispname = list(map(lambda x: x.strip().casefold(), list_prod_dispname))
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
text_embs = model.encode(list_prod_dispname[:n_rows], show_progress_bar=True)
with open("models/model_fashion.pickle", "wb") as file:
    pickle.dump(text_embs, file)