import time,sys
from app.models import Video
from app import db
import sqlite3
import os
from SingleEvaluation import get_evaluation

#def get_video_per_id(id):
#    print(id)
#    my_video = Video.query.get_or_404(id)
#    return my_video

print ('argument 1', sys.argv[1])
print ('argument 2', sys.argv[2])

mi_id=int(sys.argv[2])

evaluation=int(get_evaluation(sys.argv[1],mi_id))
#cursor.execute("SELECT title FROM video WHERE id=?",(mi_id,))
print('evaluation', evaluation)

ruta_db = os.path.join('instance', 'ftpredictor.db')
cnx = sqlite3.connect(ruta_db)
cursor = cnx.cursor()

if evaluation==-1:
    cursor.execute("UPDATE video SET state=2, prediction=? WHERE id=?", (evaluation,mi_id,))
else:
    cursor.execute("UPDATE video SET state=1, prediction=? WHERE id=?", (evaluation,mi_id,))

# 6. -COMMIT CHANGES! (mandatory if you want to save these changes in the database)
cnx.commit()


# 7.- Close the connection with the database
cnx.close()