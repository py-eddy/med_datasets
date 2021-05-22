CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�j~�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <u       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @F333333     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v{\(�     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��
=   max       ;ě�       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�|	   max       B/�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��/   max       B/�>       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�*�   max       C��"       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B��   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�!       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�\(��   max       ?�҈�p:�       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <u       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @F(�\)     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v{\(�     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?��A��t     @  [L   
   	            	            `   7         :   =                "               	   !      
      &   '   	            <                              $                  	                     
                            N�dOqb�N�:,OF�yOk�N���N>�N�&�N<�VP� P��O�̟N��O��PD�5O��nP��Ng�RN�`P>P�O��N��O���N���N��PPO�oNl�!O|3OP �O�l�O�O]9ONW��N�y�N`�cP:3�N�rN?�fN���N`��N/��N��BP;�N`0LN��%O�P�Nf��N�DN�3�M���OYVN��N��zOn��O���N�| N��PN���O���O�(�NPs�Nl�5OW�Nn��N�كNe�O.�w<u<t�<o;�o;D��;o��o���
�ě��ě���`B�#�
�49X�49X�49X�D���u�u��o��C���t����㼛�㼬1��1�ě��ě���/��/��/��/��`B��h��h��h�����o�o�o�+�C��\)�t���P��w�0 Ž49X�<j�@��L�ͽL�ͽP�`�]/�]/�e`B�m�h�u�y�#��%��o��o��o��\)��\)���P���-�� �<BOP[][OB@<<<<<<<<<<��������������������������������������'06SXPMIE<0.414#��������������������jmtz�����|zynmfejjjjR[histthe[RRRRRRRRR��""����������������������#0{������{U<0��lt�������������ojkhl!#/<HQWNKE9/#
��������������������������������������������&8:6*�����������������������������������������
##+)#










"#$/:<@><1/-#


z�����������������wz������������������������������������������������������������^ahmyz���zmma\XZ^^^^��������������������5B[g���������g[5)rtu�������tkrrrrrrrrEN[^cgmtz���tlgNIGCE�������������������������������������� 	*6CGPTWXSWOC ��z|����������������zz55BCN[b[XNB5555555559<HUWYWURH<:8:999999��������������������/6O[t�������wh[O90-/chtw{uthb_cccccccccc���

	������������HOP[ghjhe[YOGEHHHHHHMO[afhhh[OLJMMMMMMMM;<DHRUZ[UHD<;;;;;;;;ABFN[agg[NMGB@AAAAAA:<HUn���������zU<21:������������������������������������������������������������%)6;<:76/)%$%%%%%%%%�������������������� )/69BFIB62)$��������������������������������������������������������������������������������tz}���������������zt��������������������!#/1<CEHIHB<:/.##!!!��������������������gt������������tkid[gTUaaafbghfaa\UPLMPTT$).5=Ngt~tg\SNB5)#"$������������� 
 �����������))(#t�������������|trnkt����� ��������������#//4212/.#
��������������������:<HSUahnnd[UMH<2003:�a�Z�`�a�m�n�u�z�p�n�a�a�a�a�a�a�a�a�a�a�4�+�(�#�!�0�A�M�Z�`�s������u�o�d�Z�M�4�'������'�4�:�@�M�Y�Z�b�Y�M�@�4�'�'�������������ʼ�������
������̼����
���������
��/�<�?�M�U�V�T�H�<�/�#��
àÛÓÍÓàìù������������ùìàààà���������ûŻлջһлȻû����������������� �����������"�!�������z�x�m�l�j�m�z���������~�z�z�z�z�z�z�z�z���������s�a�X�-�)�5�Z��������������� ���s�m�s�h�i�Z�T�Z�s�����ʾ������׾��s�<�/�#��
�����/�<�H�U�a�h�j�U�K�H�<���������������������	�����	�������������'�,�M�f�l�r�v�x�u�i�Y�@�4�����������������!�:�G�W�X�M�?�.��ּʼ��y�s�o�`�Y�]�`�y�����������������������y�5�"���5�A�Z�g�s�{�����������g�Z�N�A�5�нŽннݽ�����ݽннннннннн��6�,�)�)�)�,�6�9�B�O�U�O�N�O�P�O�K�B�6�6������ķĵĿ������#�I�b�t�u�n�b�I��������y�f�Q�K�R�m�y�����������Ŀ˿ϿʿĿ�����������������������������������������x�o�j�l�x���������ûл׻ݻܻлû����s�p�s�v�������������������������s�s�s�s�H�H�<�9�1�<�H�U�\�a�h�k�c�a�X�U�H�H�H�H�ʾ����Ǿ׾��������*�'�3�9�*�+�#����¿³²±²·¿����������¿¿¿¿¿¿¿¿�����*�1�6�?�C�M�O�P�X�\�]�K�C�6�*��;�/�3�9�D�G�T�`�m�s���������y�v�m�T�G�;�Ŀ����������������Ŀѿݿ����������ݿĿ"��	����������	��"�.�G�T�V�R�M�G�;�"����������������������������������������������������������� ����������������������������������������������������������ìääæìù������ùìììììììììì���w�t�v�o�d�g�s������������	�������������'�3�4�4�3�'�������������������������������������������������Źù����������ùɹϹ۹ܹ�ܹϹùùùùùúY�T�M�Y�e�o�r�r�}�x�r�e�Y�Y�Y�Y�Y�Y�Y�Y�(�'�����(�5�;�:�5�*�(�(�(�(�(�(�(�(ŇŅŇŊŐŔŠŭŲŮŭŠŗŔŇŇŇŇŇŇ�a�T�<�3�-�:�T�m�z�������������������z�aƁ�~�xƁƎƚƧƫƧƝƚƎƁƁƁƁƁƁƁƁ���������������������������������������������x�e�_�V�\�l�x�������ƻʻɻû��������M�B�A�M�Y�f�r�|�r�p�f�Y�M�M�M�M�M�M�M�M���������Ŀǿѿݿ޿ݿҿѿſĿ���������������������������������������������������������&�)�0�)������������0�,�$��$�0�=�F�I�V�Y�b�i�h�b�V�S�I�=�0�ּмʼ¼��������ʼӼּ��������ּ��� �����"�(�0�5�<�5�,�(�������@�<�3�*�,�2�@�L�Y�e�~�������~�e�O�L�B�@���������������ɺֺ�����������ֺɺ�E�E�E�E�FFFF"F$F&F1F3F1F'F$FFFE�E��B�@�>�B�O�[�h�tāčĕėčā�t�h�[�O�B�B�y�{�q¦¿����������������������������	��"�"�.�7�.�"��	�����U�N�H�B�<�0�+�3�<�A�O�a�n�ÇÃ�{�n�a�U�\�H�G�<�S�`�l�y�����нٽ׽н½������l�\��
�����!�,�.�9�.�!������������������&�������������ĺĵĸıĳĿ��������������������������ĺ�ֺԺɺźźɺպֺ������������ֺֺֺ�E7E7E7E:ECEDEPE\EiEuExEuEnEiEbE\EPECE7E7�����������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� b X U f B R n _ q X ? 2 V H a > & 4 T c 2 7 1 H E z 7 ^ 7 % 7 H k 2 A K % : L 3 4 _ , L + 5 T � L J D ? F a F [ t P x _ O 9 ] $ e K : D  Q        �  �  �  �  �  ]  �  S    �  �  [  Q  y  �  �    �  {    �  �  j  |  �    �  �  �  �  �  �  )  d  �  k  P  �  ;    �  �  �  �  �    /    �  L  "  �  `  �    _  "  h  �  �  �    v  ~;ě�$�  �T���t���9X��`B�ě��D���#�
����y�#��P���㽍O߽�t���h�@���t���j�L�ͽt����,1�������ixռ���P�Y������+��P�+�C��t���Q��P�C��8Q�0 Ž���w�u�#�
�D�������aG��@��ixս]/��7L�q�������hs�����}󶽗�P��E���hs���T��^5������\)��-�����
=��1���`B��BKxB��B%�TB�+A�zBB��Bf^B�AB&�KB iBB�aB��B ��B-òB��B*�B[;B؂B��B+
�B �B ��A�|	B!�LB��B��B�#B��BGB/�BbbB	5B8�B0iBO�B�GBb�B5B)�BZNBf�B�DB EB�RB��B�B�FBtCBxYBmBB
BB�B(�Bp{BB
W�B�B=wBH�Bk<B��B
��B#F�B�BKtBp�B��B<}B�VB&ABB�/A�~�B�6B(B=�B&��B��B��B�0B �B.�B�B6�BTSB��B<NB*�nB �jB }hA��/B!�rBzB�WB��B�BB5B/�>BPMBL�B@B�_B@�B��B��B</B@IBC^B�^B��B?�B�dB)UB�~B7�B�~B��B�@B*rBE�B:�BH~BK<B��B
@�B��B:�B@�B��B�&B
BB#FOB��B=�BĩA�)�A>Nt@϶n@�m�A��}A��H@���A�F4A�*�A��=AN�A�d1A��p@�e�AdApTA�#�A*�A�A|A�bAqYBY�@���A��2A�/EAX
6A�	gB d�Ag�xA{%VA_m2A�o^A��-A��zA��KA��M?���A��>�*�?�v�A��#A��A�e�B�aA��G@���@ڪ_Ay��@�kAՓ/B
�A�A�q?׬@:E�C��"Aۖ�A�G�A\��AŪvA��AVyA0��A�E�@BTcC��B�C�#�A�~A;z[@ʥ@�+A��"A��@��A�{�A�k�A�pfAO@wA�\A�8�@�A�AocA���A*� A��A�1AoПB��@���A�X�AŃ�AYђA��[B �"Ah��Az��A]�A�}>A���A��A�S A�~,?��{A�wp>B��?���A�~�A�A���B?MA��@��@ز�A{�@�uA�|�B��@�,TA��?�jP@;�C���A�`�A��A]�A�|.A��A A/�*A�z@8�JC���BƤC�(�   
   
            
            `   7      	   ;   >      !         "               
   "            '   (   
            =                              %                  
                                                             !                  A   9            5      #         1   '      !         3               !               /                     -                                             +         #                                                   7   1            +               1   '      !                                       %                     -                                             +                              N�dONQ�N�WtO)|�Ok�N���N>�N�&�N<�VP�!Py�pNB��N��OA��P@O��nO�H�Ng�RN�`P>P�O��N��O���N���N��PO�ƖNl�!N�_O8�iO=��O��jO]9ONW��N�y�N`�cOۅwN�rN?�fNnY�N`��N/��N��BP;�N`0LN+�O|�N=��N�DN�3�M���OYVN��N��zOZo�Oz��N�| N��PN�ДO���O��pNPs�Nl�5OW�Nn��N���Ne�O.�w  E  E  /  �  F  �  L  
  :  �  a  �  Z  �  z  �  [    �  `  )  H    :  Z  z  >  (  �  �  =  �  x  G  �  �  �  �  :  �  �  �  �  �  5  ,    h  �  �  6  �  �  |  �  �  �  �    �  -  �    �    	h  �  5<u<o;ě�;D��;D��;o��o���
�ě���t��D����`B�49X���ͼě��D����/�u��o��C���t����㼛�㼬1��1���ě���`B��h�C��C���`B��h��h��h�D�����o�C��o�+�C��\)�t��#�
�8Q�8Q�49X�<j�@��L�ͽL�ͽP�`�aG��aG��e`B�m�h�u�}󶽁%��O߽�o��o��\)��\)���㽝�-�� �<BOP[][OB@<<<<<<<<<<��������������������������������������#04QWNKI@<835,#��������������������jmtz�����|zynmfejjjjR[histthe[RRRRRRRRR��""�������������������#0b{�����{U<0	 �t��������������tnkmt#//30/#������������������������������������������ 00)����������������������������������	����������
##+)#










"#$/:<@><1/-#


z�����������������wz������������������������������������������������������������^ahmyz���zmma\XZ^^^^��������������������3;BNZdr{��tg[NB:2.03rtu�������tkrrrrrrrrMN[\agktt}tg[VNJHGMM������������������������������������� *6CKOPOIC6*�� z|����������������zz55BCN[b[XNB5555555559<HUWYWURH<:8:999999��������������������57BO[t�}}~|th[OF@75chtw{uthb_cccccccccc���

	������������IOS[chhhc[UOJFIIIIIIMO[afhhh[OLJMMMMMMMM;<DHRUZ[UHD<;;;;;;;;ABFN[agg[NMGB@AAAAAA:<HUn���������zU<21:������������������������������������������������������������&)6:;9665)&%&&&&&&&&�������������������� )/69BFIB62)$��������������������������������������������������������������������������������������������������{���������������������!#/1<CEHIHB<:/.##!!!��������������������gt������������tkid[gTUafgfa`[UQMNPTTTTTT$).5=Ngt~tg\SNB5)#"$�����	��������� 
 �����������))(#t�������������|trnkt����� ��������������#./31/.#��������������������:<HSUahnnd[UMH<2003:�a�Z�`�a�m�n�u�z�p�n�a�a�a�a�a�a�a�a�a�a�4�2�(�'�(�5�A�M�Z�b�s������s�k�`�Z�M�4�'����� �'�4�7�@�M�S�Y�[�Y�M�@�4�'�'�����������ʼ�����������ּʼ��������
���������
��/�<�?�M�U�V�T�H�<�/�#��
àÛÓÍÓàìù������������ùìàààà���������ûŻлջһлȻû����������������� �����������"�!�������z�x�m�l�j�m�z���������~�z�z�z�z�z�z�z�z���w�h�_�E�>�A�Z�������������������������{�y�o�p�_�Z�[�s�����ʾ߾����������{�H�A�<�3�7�<�H�U�V�Y�U�L�H�H�H�H�H�H�H�H���������������������	�����	��������4�'����$�'�4�@�J�M�Y�f�p�r�m�f�_�Y�4�������ļ����.�:�K�L�I�B�:�.����ּ��y�s�o�`�Y�]�`�y�����������������������y�A�5�+�&�)�5�A�N�Z�g�u���������s�g�Z�N�A�нŽннݽ�����ݽннннннннн��6�,�)�)�)�,�6�9�B�O�U�O�N�O�P�O�K�B�6�6������ķĵĿ������#�I�b�t�u�n�b�I��������y�f�Q�K�R�m�y�����������Ŀ˿ϿʿĿ�����������������������������������������x�o�j�l�x���������ûл׻ݻܻлû����s�p�s�v�������������������������s�s�s�s�H�H�<�9�1�<�H�U�\�a�h�k�c�a�X�U�H�H�H�H�׾ʾǾ̾׾����	�� ������	����¿³²±²·¿����������¿¿¿¿¿¿¿¿�����*�3�6�B�C�O�U�\�S�O�J�C�6�*���;�2�4�;�;�E�G�T�`�m�q�����y�t�m�`�T�G�;�ݿĿ��������������Ŀѿݿ����������ݿ"��	���������	��"�.�;�B�F�G�G�A�;�.�"����������������������������������������������������������� ����������������������������������������������������������ìääæìù������ùìììììììììì�����������������������������������������'�3�4�4�3�'�������������������������������������������������Źù����������ùĹϹعܹ�ܹϹùùùùùúY�T�M�Y�e�o�r�r�}�x�r�e�Y�Y�Y�Y�Y�Y�Y�Y�(�'�����(�5�;�:�5�*�(�(�(�(�(�(�(�(ŇŅŇŊŐŔŠŭŲŮŭŠŗŔŇŇŇŇŇŇ�a�T�<�3�-�:�T�m�z�������������������z�aƁ�~�xƁƎƚƧƫƧƝƚƎƁƁƁƁƁƁƁƁ���������������������������������������޻����x�k�l�w�������������ûǻƻû»������M�E�C�M�Y�f�r�s�r�n�f�Y�M�M�M�M�M�M�M�M���������Ŀǿѿݿ޿ݿҿѿſĿ���������������������������������������������������������&�)�0�)������������0�,�$��$�0�=�F�I�V�Y�b�i�h�b�V�S�I�=�0�ּмʼ¼��������ʼӼּ��������ּ��� �����"�(�0�5�<�5�,�(�������3�/�+�-�3�@�L�Y�e�~�����~�|�e�_�N�L�@�3�������������ɺֺ̺�����������ֺɺ�E�E�E�E�FFFF"F$F&F1F3F1F'F$FFFE�E��B�@�>�B�O�[�h�tāčĕėčā�t�h�[�O�B�B�y�{�q¦¿������������������������	�� �"�.�6�.�"��	�������������U�N�H�B�<�0�+�3�<�A�O�a�n�ÇÃ�{�n�a�U�l�`�T�O�S�Z�`�l�y�������ȽȽ��������y�l��
�����!�,�.�9�.�!������������������&�������������ĺĵĸıĳĿ��������������������������ĺ�ֺԺɺźźɺպֺ������������ֺֺֺ�ECE8E;ECEGEPE\EfEiEmEiEaE\EPECECECECECEC�����������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� b Z R b B R n _ q E A - V 9 T > ( 4 T c 2 7 1 H E G 7 \ 2  7 H k 2 A K % : D 3 4 _ , L 0 # Q � L J D ? F Y B [ t P g _ E 9 ] $ e 2 : D  Q  �  �  �  �  �  �  �  �  N  ,  ]    �  �  [  !  y  �  �    �  {    �  ]  j  !  �  �  4  �  �  �  �    )  d  q  k  P  �  ;    8  �  d  �  �    /    �  �  �  �  `  �  �  _  P  h  �  �  �  �  v  ~  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  E  @  :  -      �  �  �  �  �  T  �  y  D    �  �  a  &  C  D  E  B  >  7  /  "      �  �  �  �  �  {  Z  >  C  H    #  /  )    �  �  �  �  `  A    �  �  T  	  �  b    �  �  �  �  �  �  �  �  �  �  �  �  u  f  \  9    �  x  &   �  F  B  8  ,    	  �  �  �  �  d  6  �  �  V  �  �  B  �  #  �  �  �  �  �  �  �  �  �  �  �  �  y  o  `  J    �  �  p  L  L  L  L  K  H  D  @  6  #    �  �  �  �  g  4  �  �  �  
  �  �  �  �  �  �  �  �  �  �  �  y  e  P  :  %  �  �  �  :  /  %        �  �  �  �  �  �  �  i  P  5     �   �   �  M  �  �  �  �  �  k  K    �  �  E  �  �  K  �  [  �  �  &  :  X  a  R  =  )      �  �  �  j    �  s  $  �  H  �   �  O  _  ^  [  [  _  d  r  �  �  �  �  �  �  �  v  8  �  \  �  Z  Y  Y  S  J  A  7  .         �  �  �  ~  ]  ?    �  �    U  �  �  �  �  �  �  \  $  �  �  q    �     �  �  7  �  #  H  i  w  z  v  l  R  ,  �  �  ~  I    �  E  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  P    �  u    �       0  @  K  T  Z  Y  Q  C  '  �  �  m    �  n  %  m    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  h  X  K  >  *    �  �  �  �  �  �  c  ?    `  >    
         "  -  3  :  :  "    �  �  b    �  �  )    �  �  �  �  �  h  Q  @  0          �  �  �  �  O  	  H  B  <  7  2  +       �  �  �  �  �  ]  4    �  �  k  9    �  �  �  �  �  �  �  �  �  |  ]  B  *  .  �  �  �  i  6  :  /  %        �  �  �  �  �  �  �  �  �  ~  Z  0     �  Z  Y  X  H  6    �  �  �  �  i  >    �  �  �  f  B  )    �  �  �  �    A  ^  ]  w  d  ?    �  �  ]  9  �  �  q    >  9  5  0  ,  (  $  "  "  !           �  �  �  �  }  d      (      	  �  �  �  �  �  �  |  Z  7    �  �  �  �  �  �  �  �  �  y  n  l  S  2    �  �  �  �  f  A    �  �  K  �  �  �  �  �  }  k  T  2  
  �  �  ~  >  �  \  �  �  �  �    /  ;  =  ;  5  ,    �  �  �  n  :  �  �  >  �  �    �  �  �  v  h  V  B  *    �  �  �  �  t  n  c  K  1      �  x  n  c  X  M  B  7  ,  !    
  �  �  �  �  �  �  �  �  �  G  E  C  A  ?  >  =  <  9  1  *  #    �  �  �  �  �  �  �  �  x  m  c  W  L  A  6  ,  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  @    �  w    �    g  8   V  �  �  �  �  �  �  �  �  q  a  L  0    �  �  �  j  >    �  �  �  �  �  �  �  �  �  {  m  f  h  i  j  k  m  n  o  p  r  -  3  8  9  6  1  )      �  �  �  �  �  z  U  %  �    �  �  �  �  �  �  p  ]  I  1    �  �  �  u  G    �  �  �  P  �  �  }  w  o  `  P  @  /      �  �  �  �  �  �  m  V  ?  �  �  �  �  �  �  �  �  �  o  [  H  4  !    �  �    ;   �  �  �  �  �  �  �  �  h  3  _  [  I  +    �  �  p    �  A  �  �  �  �  �  �  u  j  ^  S  A  +     �   �   �   �   �   �   �  0  .  /  2  4  5  9  >  A  A  =  V    �  |  u  n  d  Y  M      #  +  !    �  �  �  �  n  N  )  �  �  s    �    A      	      �  �  �  �  �  �  �  �  s  Y  0  �  �      h  X  H  8  (      �  �  �  �  �  �  |  e  N  7     	   �  �  �  �  �  �  �  �  x  f  T  B  /    �  �  �  �  m  <  
  �  �  �  �  �  �  ~  w  q  h  ^  T  J  ?  4  (        �  6      �  �  �  �  �  �  �  �  ]  *  �  �  ~  <  �  �  `  �  w  a  O  >  /  !    �  �  �  �  �  �  �  �  �  �  �  i  �  s  M  $  �  �  �  �  a  4    �  �  l  0  �  �  K  �  �  `  y  r  b  N  8  !  
  �  �  �  u  J    �  �  �  @  �  C  �  �  �  �  �  �  z  d  G  &  �  �  �  f  )  �  �  R  �  v  �  q  _  N  9  $    �  �  �  W    �  �  �  }  c  A     �  �  �  �  �  w  Y  ;    �  �  �  P    �  �  H    �  �  �  �  ~  w  k  _  O  8    �  �  �  �  �  [  0  �  �  *  �   �  �  �  �  �  �  �  �  |  a  E  -          �  �  �  �  �  �  �  �  l  M  .    �  �  �  �  C  �  �  x  U  Q    �  �      %  ,  +      �  �  �  �  �  �  �  m  8  �  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  U  "  �  �  m  ,  �          �  �  �  �  �  �  �  �  �  �  �  o  Y  G  4  "  �  p  g  i  n  o  f  R  :  "  
  �  �  �  �  �  �  ]  1           �      )  *  !        �  �  �  �  �  �  �  �  �  	C  	g  	_  	T  	?  	$  	  �  �  �  O    �  n    �  G  �  I  �  �  �  r  P  -    �  �  �  R  "    �  �  �  K  	   �   �  5  
  �  �  z  L    �  �  �  L    �  �  �  r  L  &    �