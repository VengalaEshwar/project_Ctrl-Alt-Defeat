//importing the requored modules
const express = require('express');
const router = require("./routes/routes.js");

//giving express functionlaities to the app
const app = express();
let port = 5000;

//starting the server
app.listen(port,()=>
{
    console.log("Server Started at port",port);
});
//middlewares
app.use(express.json());
//passing app to routes
router(app);