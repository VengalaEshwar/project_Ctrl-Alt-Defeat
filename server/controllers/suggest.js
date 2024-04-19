exports.suggest = (req,prediction)=> {
let res ={};
if(prediction==0)
{
    res['result'] = "Negitive";
}
else
{
    res['result'] = "Positive";
    req['sugar'] =model.sugar(req.)
    
}
} 