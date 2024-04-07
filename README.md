From Photshop to go to the project
cd pythonProject/photoshop_app/

To start admin server : python manage.py createsuperuser
admin login id pass : username- manvi09 password - manvi09

To start webpage : python manage.py runserver

server : http://127.0.0.1:8000/
admin : http://127.0.0.1:8000/admin/
register : http://127.0.0.1:8000/register/
scan : http://127.0.0.1:8000/scan/


python manage.py makemigrations
python manage.py migrate
: if any database changes are there in models .py we need to run these commands
