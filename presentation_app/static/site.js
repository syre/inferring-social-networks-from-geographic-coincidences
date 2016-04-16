$(document).ready(function() {

$("#cooc-submit-button").on("click", function()
{
	var user1 = $("#user1-dropdown").find(":selected").text();
	var user2 = $("#user2-dropdown").find(":selected").text();
	window.location.href = "cooccurrences?useruuid1="+encodeURIComponent(user1)+"&useruuid2="+encodeURIComponent(user2);
});

});
