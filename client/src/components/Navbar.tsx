import { Box, Typography, AppBar, Button, Toolbar } from '@mui/material'
const Navbar = () => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            COMP309-Group2
          </Typography>
          <Button color="inherit" href='/'>Home</Button>
          <Button color="inherit" href='https://data.torontopolice.on.ca/datasets/a89d10d5e28444ceb0c8d1d4c0ee39cc_0/explore?location=15.927605%2C-28.347773%2C2.59' target='_blank'>Data Exploration</Button>
          <Button color="inherit" href='https://github.com/TariqueJemison01/Toronto-Bike-Theft-Predictive-Model' target='_blank'>Github</Button>
        </Toolbar>
      </AppBar>
    </Box>
  )
}

export default Navbar