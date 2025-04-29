import time,sys
from app.models import Video
from app import db
import sqlite3

def get_video_per_id(id):
    print(id)
    my_video = Video.query.get_or_404(id)
    return my_video

cnx = sqlite3.connect('instance/ftpredictor.db')
cursor = cnx.cursor()

for i in range(2):
    print(f"Ejecutando tarea {i}")
    time.sleep(1)

print ('argument 1', sys.argv[1])
print ('argument 2', sys.argv[2])

mi_id=int(sys.argv[2])

#cursor.execute("SELECT title FROM video WHERE id=?",(mi_id,))

cursor.execute("UPDATE video SET state=1 WHERE id=?",(mi_id,))

# 6. -COMMIT CHANGES! (mandatory if you want to save these changes in the database)
cnx.commit()


# 7.- Close the connection with the database
cnx.close()