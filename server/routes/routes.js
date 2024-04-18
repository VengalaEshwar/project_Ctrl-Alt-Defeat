const controller = require("../controllers/controller.js");
module.exports = (app) => {
    app.get("/",controller.start);
    app.get("*",controller.error);
}