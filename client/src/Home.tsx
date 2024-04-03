import {
    Box,
    Button,
    FormControl,
    FormLabel,
    Grid,
    TextField,
    Typography,
} from "@mui/material";
import { DatePicker, TimePicker } from "@mui/x-date-pickers";
import { useState, MouseEvent } from "react";
type PredictiveModelSelection = "lr" | "dt" | "rf";
const Home = () => {
    const [model, setModel] = useState<PredictiveModelSelection>();
    const handleModelSelect = (e: MouseEvent<HTMLButtonElement>) => {
        const modelSelected = e.currentTarget.id as PredictiveModelSelection;
        console.log(modelSelected);
        setModel(modelSelected);
    };
    return (
        <main>
            <Box
                component={"form"}
                margin={"auto"}
                marginTop={5}
                marginBottom={2}
                width={700}
                boxShadow={5}
                padding={1}
            >
                <Typography
                    textAlign={"center"}
                    marginBottom={3}
                    fontSize={20}
                    sx={{ fontWeight: "bold" }}
                >
                    Predicting Bike Status
                </Typography>
                <FormControl sx={{ display: "block", marginBottom: 1 }}>
                    <FormLabel>Select predictive model:</FormLabel>
                    <Grid
                        width={"90%"}
                        margin={"auto"}
                        justifyContent={"center"}
                        container
                        columnSpacing={1}
                        columns={3}
                        marginTop={1}
                    >
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="lr"
                                variant={model === "lr" ? "outlined" : "text"}
                            >
                                Logistic Regression
                            </Button>
                        </Grid>
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="rf"
                                variant={model === "rf" ? "outlined" : "text"}
                            >
                                Random Forest
                            </Button>
                        </Grid>
                        <Grid item xs={1}>
                            <Button
                                fullWidth
                                onClick={handleModelSelect}
                                id="dt"
                                variant={model === "dt" ? "outlined" : "text"}
                            >
                                Decision Tree
                            </Button>
                        </Grid>
                    </Grid>
                </FormControl>
                <Grid
                    container
                    margin={"auto"}
                    width={"100%"}
                    columns={2}
                    spacing={1}
                >
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Primary Offense"
                            fullWidth
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Premises type"
                            fullWidth
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <DatePicker label="Occur Date" sx={{ width: "100%" }} />
                    </Grid>
                    <Grid xs={1} item>
                        <TimePicker
                            label="Occur Hour"
                            sx={{ width: "100%" }}
                            format="HH:00"
                            views={["hours"]}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <DatePicker
                            label="Report Date"
                            sx={{ width: "100%" }}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TimePicker
                            label="Report Hour"
                            sx={{ width: "100%" }}
                            format="HH:00"
                            views={["hours"]}
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="text" label="Division" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField
                            type="text"
                            label="Location Type"
                            fullWidth
                        />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="text" label="Bike Make" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="text" label="Bike Model" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="text" label="Bike Type" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="text" label="Bike Color" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="number" label="Bike Cost" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="number" label="Longitude" fullWidth />
                    </Grid>
                    <Grid xs={1} item>
                        <TextField type="number" label="Latitude" fullWidth />
                    </Grid>
                    <Grid
                        xs={1}
                        item
                        display={"flex"}
                        alignItems={"center"}
                        justifyContent={"center"}
                    >
                        <Button sx={{ margin: "auto", width: "50%" }} variant="contained" color="secondary">
                            Reset
                        </Button>
                    </Grid>
                </Grid>
                <Grid container columns={2} marginTop={2}>
                    <Grid item xs={1} >
                        <Button variant="contained" sx={{margin:"auto",display:"block",width:"80%"}} color="warning">Auto Fill</Button>
                    </Grid>
                    <Grid item xs={1}>
                        <Button variant="contained" type="submit" sx={{margin:"auto",display:"block",width:"80%"}}>Submit</Button>
                    </Grid>
                </Grid>
            </Box>
        </main>
    );
};

export default Home;
