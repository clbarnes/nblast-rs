const path = require('path');
// const HtmlWebpackPlugin = require('html-webpack-plugin');

const WasmPackPlugin = require('@wasm-tool/wasm-pack-plugin')

module.exports = {
  mode: 'development',
  entry: './nblast.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    globalObject: 'this',
    filename: "nblast.js",
    library: {
      name: "nblast",
      type: "umd"
    }
  },
  // devServer: {
  //     open: true,
  //     host: '0.0.0.0',
  // },
  plugins: [
    // new HtmlWebpackPlugin({
    //     template: 'index.html',
    // }),
    new WasmPackPlugin({
        crateDirectory: __dirname,
        outDir: path.join(__dirname, './pkg'),
        outName: "nblast_wasm",
    })
  ],
  // module: {
  //     rules: [
  //         {
  //             test: /\.(eot|svg|ttf|woff|woff2|png|jpg|gif)$/,
  //             type: 'asset',
  //         },
  //     ],
  // },
  experiments: {
    asyncWebAssembly: true,
  }
};
