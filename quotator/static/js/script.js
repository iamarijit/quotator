var visible = false;

function copyToClip() 
{
    var r = document.createRange();
    var ele = document.getElementById("quote")
    var selection = window.getSelection();

    r.selectNode(ele);
    selection.removeAllRanges();
    selection.addRange(r);
    console.log(document.execCommand('copy'));
    selection.removeAllRanges();

    toggleCopyMessage();
    setTimeout(toggleCopyMessage, 2000);
}

function toggleCopyMessage()
{
    var ele = document.getElementById("copymessage");
    if (visible == true) 
    {
        ele.style.display = "none";
        visible = false;
    }
    else
    {
        ele.style.display = "block";
        visible = true;
    }
}