# Deploying Web Apps to Heroku



## Prerequisites:
- Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)



## Initial Steps
Create an account on the [Heroku Homepage](https://www.heroku.com/) if you don't have one already. 

Then log in to your Heroku account from the command line: 
```bash
heroku login
```

Clone your git repo to your local machine and navigate to it
```
git clone https://www.github.com/raghavrajmittal/myapp
cd myapp
```



## Create a Heroku remote
Git remotes are versions of your repository that live on other servers. You deploy your app by pushing its code to a special Heroku-hosted remote that’s associated with your app.

#### For a new Heroku App
```
heroku create
```
This creates a Heroku remote. To check if it was created, run ```git remote -v``` and you should see both the remotes listed, origin and heroku.

#### For an existing Heroku APP
If you have already created your Heroku app, you can sipmly add a remote to the local repository with the ```heroku git:remote command```. All you need is the Heroku app’s name:
 ```
heroku git:remote -a floating-shore-96106
```



## Deploy/Push to the Heroku remote
```
git push heroku master
```
Use this same command whenever you want to deploy the latest committed version of your code to Heroku.
Note that Heroku only deploys code that you push to the master branch of the heroku remote. Pushing code to another branch of the remote has no effect.

Once the application is deployed, Heroku will provide a URL you can visit to see the site. Running ```heroku open``` will open a browser window and navigate to the deployed application URL.




## Extra
By default, the Heroku CLI names all of the Heroku remotes it creates for your app 'heroku'. You can rename your remotes with the ```git remote rename``` command:
```
git remote rename heroku heroku-staging
```
