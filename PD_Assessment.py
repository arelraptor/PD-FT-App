import time,sys
from app.models import Video
from app import db
import sqlite3
import os
from SingleEvaluation import get_evaluation


print ('argument 1: ', sys.argv[1])
print ('argument 2: ', sys.argv[2])

mi_id=int(sys.argv[2])

evaluation=int(get_evaluation(sys.argv[1],mi_id))

print('Evaluation: ', evaluation)

ruta_db = os.path.join('instance', 'ftpredictor.db')
cnx = sqlite3.connect(ruta_db)
cursor = cnx.cursor()

if evaluation==-1:
    cursor.execute("UPDATE video SET state=2, prediction=? WHERE id=?", (evaluation,mi_id,))
else:
    cursor.execute("UPDATE video SET state=1, prediction=? WHERE id=?", (evaluation,mi_id,))

# COMMIT CHANGES! (mandatory if you want to save these changes in the database)
cnx.commit()


# Close the connection with the database
cnx.close()