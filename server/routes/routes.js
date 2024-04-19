const controller = require("../controllers/controller.js");
module.exports = (app) => {
    app.get("/project",controller.start);
    app.post("/project/postData",controller.postData);
    app.get("*",controller.error);
}