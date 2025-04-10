$(document).ready(function () {
    $('#predict-btn').click(function () {
        const data = {
            soil_ph: $('#soil_ph').val(),
            temperature: $('#temperature').val(),
            rainfall: $('#rainfall').val(),
            fertilizer: $('#fertilizer').val(),
            pesticide: $('#pesticide').val()
        };

        $.ajax({
            type: 'POST',
            url: '/predict_ajax',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function (response) {
                $('#recommended-crop').text(response.crop);
                updateChart(response.probabilities, response.labels);
            },
            error: function () {
                alert("Something went wrong with prediction!");
            }
        });
    });

    function updateChart(probabilities, labels) {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        if (window.probChart) window.probChart.destroy(); // Reset if chart already exists

        window.probChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability',
                    data: probabilities,
                    backgroundColor: 'rgba(0, 128, 0, 0.6)'
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
});
