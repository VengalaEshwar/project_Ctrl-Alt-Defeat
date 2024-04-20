//importing the requored modules
const express = require('express');
const cors = require('cors');
const router = require("./routes/routes.js");

//giving express functionlaities to the app
const app = express();
let port = 5000;
app.use(cors());
//starting the server
app.listen(port,()=>
{
    console.log("Server Started at port",port);
});
//middlewares
app.use(express.json());
//passing app to routes
router(app);