from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route('/abstract_layout')
def abstract_layout():
    return render_template('abstract_layout_page.html')


@app.route('/layout', methods=['GET', 'POST'])
def layout():
    if request.method == 'POST':
        num_red_nodes = 20
        num_orange_nodes = int(request.form.get('numOrangeNodes', 1))

        return jsonify({
            'numRedNodes': num_red_nodes,
            'numOrangeNodes': num_orange_nodes,
        })
    
    return render_template('layout_page.html')


if __name__ == '__main__':
    app.run(debug=True)