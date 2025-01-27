import React, { Component } from "react";
import Canvas from "./components/Canvas";
import Dropdown from "./components/Dropdown";
import { rotationMatrix } from "mathjs";
import * as ut from "./utils";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      zoom: 300,
      pointSize: 18,
      projectionTypeOptions: [
        { value: "stereo", label: "Stereographic" },
        { value: "ortho", label: "Orthographic" },
        { value: "cylindre", label: "Cylindrical" },
      ],
      projectionType: "stereo",
      rotMat: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      pointDataSpherical: null,
      pointDataCanvas: null,
      pointDataXY: null,
    };
  }

  fetchDataSet = () => {
    fetch("http://localhost:8000/thumbnails/")
      .then((r) => r.json())
      .then((pointDataSpherical) => {
        this.setState({ pointDataSpherical }, this.projectData);
      });
  };

  setProjectionType = (projectionType) => {
    this.setState({ projectionType }, this.projectData);
  };

  handleMouseWheelEvent = (e) => {
    const zoom = Math.max(150, this.state.zoom + e.deltaY);
    const pointSize = Number.parseInt(0.06 * zoom, 10);
    this.setState({ zoom, pointSize }, this.handleZoomLevelChanged);
  };

  handleZoomLevelChanged = () => {
    const { zoom, canvasSizeX, canvasSizeY, pointDataXY } = this.state;
    const pointDataCanvas = [];
    for (const [, { id, x, y, thumbnail }] of Object.entries(pointDataXY)) {
      const xResized = x * zoom + canvasSizeX / 2;
      const yResized = y * zoom + canvasSizeY / 2;
      if (
        xResized >= 0 &&
        yResized >= 0 &&
        xResized <= canvasSizeX &&
        yResized <= canvasSizeY
      ) {
        pointDataCanvas.push({ id, xResized, yResized, thumbnail });
      }
    }
    this.setState({ pointDataCanvas });
  };

  projectData = () => {
    const { pointDataSpherical, projectionType } = this.state;
    const pointDataXY = [];
    if (pointDataSpherical) {
      for (const [, { id, theta, phi, thumbnail }] of Object.entries(pointDataSpherical)) {
        let x, y;
        switch (projectionType) {
          case "ortho":
            [x, y] = ut.orthoProjection([theta, phi]);
            break;
          case "stereo":
            [x, y] = ut.stereoProjection(ut.spher2cart([theta, phi]));
            break;
          case "cylindre":
            [x, y] = ut.equalAreaProjection([theta, phi]);
            break;
          default:
            [x, y] = [0, 0];
        }
        pointDataXY.push({ id, x, y, thumbnail });
      }
      this.setState({ pointDataXY }, this.handleZoomLevelChanged);
    }
  };

  getXYZ = (clickX, clickY) => {
    const { zoom, canvasSizeX, canvasSizeY, projectionType } = this.state;
    const XY = [
      (clickX - canvasSizeX / 2) / zoom,
      (clickY - canvasSizeY / 2) / zoom,
    ];
    let xyz;
    switch (projectionType) {
      case "ortho":
        xyz = ut.spher2cart(ut.invOrthoProjection(XY));
        break;
      case "stereo":
        xyz = ut.invStereoProjection(XY);
        break;
      case "cylindre":
        xyz = ut.spher2cart(ut.invEqualAreaProjection(XY));
        break;
      default:
        xyz = [];
    }
    return xyz;
  };

  handlePanViewEvent = (e) => {
    const { moving, xyzOrigin, pointDataSpherical, rotMat } = this.state;
    if (moving) {
      const micro = document.getElementById("tooltip");
      if (micro != null) {
        micro.classList.add("invisible");
      }

      const xyz = this.getXYZ(e.clientX, e.clientY);

      if (ut.arraysEqual(xyz, xyzOrigin)) {
        return;
      }

      const cvNormed = ut.crossVectorNormed(xyzOrigin, xyz);
      const angle = ut.angleBetweenVectors(xyz, xyzOrigin);
      const matrixR = rotationMatrix(angle, cvNormed);
      const newRotMat = ut.matrixMul(matrixR, rotMat);

      const newPointDataSpherical = [];
      for (const [, { id, theta, phi, thumbnail }] of Object.entries(pointDataSpherical)) {
        let newTheta, newPhi;
        [newTheta, newPhi] = ut.cart2spher(
          ut.matrixRot(matrixR, ut.spher2cart([theta, phi])),
        );
        newPointDataSpherical.push({
          id: id,
          theta: newTheta,
          phi: newPhi,
          thumbnail: thumbnail,
        })
      }

      this.setState(
        {
          pointDataSpherical: newPointDataSpherical,
          matrixR: matrixR,
          xyzOrigin: xyz,
          rotMat: newRotMat,
        },
        this.projectData,
      );
    }
  };

  downListener = (e) => {
    document.getElementById("canvas").style.cursor = "grabbing";
    const xyz = this.getXYZ(e.clientX, e.clientY);
    this.setState({ moving: true, xyzOrigin: xyz });
  };

  upListener = (e) => {
    e.preventDefault();
    document.getElementById("canvas").style.cursor = "grab";
    this.setState({ moving: false });
  };

  adjustCanvasSize = () => {
    const canvas = document.getElementById("canvas");
    const rect = canvas.getBoundingClientRect();
    const canvasSizeX = rect.width;
    const canvasSizeY = rect.height;
    this.setState({ canvasSizeX, canvasSizeY }, this.projectData);
  };

  setupEventListeners = () => {
    window.addEventListener("resize", this.adjustCanvasSize);

    document
      .getElementById("canvas")
      .addEventListener("wheel", this.handleMouseWheelEvent, { passive: true });

    document
      .getElementById("canvas")
      .addEventListener("pointerdown", this.downListener);

    document
      .getElementById("canvas")
      .addEventListener("pointermove", this.handlePanViewEvent);

    document
      .getElementById("canvas")
      .addEventListener("pointerup", this.upListener);
  };

  componentDidMount = () => {
    this.fetchDataSet();
    this.adjustCanvasSize();
    this.setupEventListeners();
  };

  render = () => {
    const { pointSize, pointDataCanvas, projectionTypeOptions } = this.state;
    return (
      <div className="App">
        <div className="canvas-wrapper">
          <Canvas pointSize={pointSize} pointDataCanvas={pointDataCanvas} />
        </div>
        <div className="side-pannel">
          <div className="dropdown-wrapper">
            <Dropdown
              label={"Projection"}
              options={projectionTypeOptions}
              actionFnct={this.setProjectionType}
              defaultValue={{ value: "stereo", label: "Stereographic" }}
            />
          </div>
        </div>
      </div>
    );
  };
}

export default App;
