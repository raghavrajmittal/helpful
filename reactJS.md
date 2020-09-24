# Create and deploy a React web app
React, created by Facebook, is a JavaScript framework for building front-end applications.

## Prerequisites:
- Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Install [VS Code](https://code.visualstudio.com/)
- Install [Node.js + npm](https://nodejs.org/en/)



## Initial Steps
The Node Package Manager (npm) keeps track of defails about the project and installs and JS packages needed.
Check whether Node and npm are installed:
```bash
node -v && npm -v
```


## Launch your first project using create-react-app
Create a new app (might take a few minutes):
<pre>
npx create-react-app <i>your-app-name</i>
</pre>

Once it's installed, navigate into the directory and start the app:
<pre>
cd <i>your-app-name</i>
npm start
</pre>

A basic React app should be visible at ```localhost:3000```.  



## Running previously written React code
If you want to run an existing app already, first install the libraries needed, and then start the app:
 
```
npm install
npm start
```


## Understanding all the files
```
build/
node_modules/
public/
src/
.gitignore
package-lock.json
package.json
README.md
```

* ```package.json```: This file contains metadata about the project, and the list of dependencies being used in the project.  
* ```public/index.html```: When the application starts this is the first page that is loaded. This will be the only html file in the entire application since React is generally written using JSX. This file has a line of code ```<div id=”root”></div>```, which is important because all the application components are loaded into this div.  
* ```src/index.js```: This is the javascript file corresponding to index.html. This file has the line of code ```ReactDOM.render(<App />,document.getElementById(‘root’));``` - which is also super important.
* ```src/App.js```: This is the file for App Component. App Component is the main component in React which acts as a container for all other components.
* ```src/*.css```: CSS files corresponding to the js files.
* ```build/```: This is the folder where the built files are stored. React Apps can be developed using either JSX, or normal JavaScript itself, but using JSX definitely makes things easier to code for the developer :). But browsers do not understand JSX. So JSX needs to be converted into javascript before deploying. These converted files are stored in the build folder after bundling and minification. In order to see the build folder Run the following command
* ```node_modules/```: Any external libraries used in the project are stored here.



## Deploying on github pages
Install the github pages package as a dev-dependecy. ```--save-dev``` automatically adds it to your package.json.
```
npm install gh-pages --save-dev
```

Add the ```homepage``` property in package.json file (at the top level, so under 'name' or under 'private'
```
"homepage": "http://raghavrajmittal.github.io/purple-theory"
```

Under 'scripts' in package.json, add the following:
```
"predeploy": "npm run build",
"deploy": "gh-pages -d build"
```

Connect your project to a github repo if you haven't already:
```
git init
git remote add origin your-github-repository-url.git
```

Deploy to github pages!
```
npm run deploy
```
Keep in mind that this creates branch named ```gh-pages``` in the GitHub repository, and hosts the live app. To confugre the root for the page, go to Settings -> Github Pages.

You can keep committing your app code to the master branch simultaneously as that does not effect your deployed version.

