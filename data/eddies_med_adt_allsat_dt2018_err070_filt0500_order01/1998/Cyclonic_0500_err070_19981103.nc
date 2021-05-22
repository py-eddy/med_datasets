CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�333333       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��I   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       ;��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @FK��Q�       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vy�Q�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @P�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�N`           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �0 �   max       �o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ʼ   max       B0�d       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�o+   max       B0J�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >P��   max       C��M       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�   max       C��v       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          K       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��I   max       P���       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?�a|�Q       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��"�   max       ��o       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�        max       @FK��Q�       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vx            P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�f            [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�������   max       ?�\��N;�     0  ^   :      1   )   $   1            F   0            ,   !                        ?               4         
   3      	                              %                  1                                       	      8      K         P%mN�.�O�P |P?�2Oǹ�N�׸O���O�iP��P��-N�f�N+\�N\�4PICiO��O^>Ni��O���OD"=OM�N��O���PC`+N��N�0�N}PZO8O�PLOG�CO�M�O'��O��O���N��N8R�N1L�NP�O=LO��OpjO�'OU�INN�SO���N`�O4�N�� Nj�YN�P	��OY��Oy*GO8�"OLZ�N��oOyͲN�/�O���OLw'O��O�Y>OS�9N�l�M��IO��"N�aO�{O���O>��N��T;��
��o�o�D���D����o��o��`B��`B�49X�49X�D���D���e`B�u��o��o��o��o��t���t���1��1��9X��j��j���ͼ�/��/��`B�������o�+�+�+��P��w��w��w��w�0 Ž8Q�P�`�e`B�q���q���q���q���q���u�y�#�y�#�y�#�y�#��+��7L��t����P�������T��� Ž� Ž�E���E��ȴ9�ȴ9����������������������2<IU_]ZXUJII?<<62222~���������������{x{~���"1"����������)BHS[h���[O)���#/<U`ffaK@</#��
#+#"
������07<HUnz||tunaUH<1.-0$*6COX\agkh\OC6*�����0b{�����I0����y�������������{xy��������������������#./2<8/#rtx|��������torrrrrr7<Tmz����������maQ?7��
#/6<@<:;/#
����=HUabnz������znaVHF=26BMMOSOJBA<64222222KOht�����������h[SNK��������������������)69BOV[db[OB)'�������

 ���")/3<HUaq}zaaU</*$"m���������������zmm7<HUUYWWUPH=<6027777���������� �������������������������������� ��������������)-2B6���������PTamz�������zmaYTRNPGUanz������zrnj`UQFG��������������������4=BO[hitohe`[OB72234����������������������������� ����������&)58?65)&&&&&&&&&&��������������������
 � ��������������������')57BNP[t�����t[B4+'Y[agkqtvyz{xutsg_[XYU[eowx���������tl[TU`cmz����������zqmea`rtw~��������trrrrrrr��������� �����������������������������fgt��������tqjgc`_afrtz���������yttsrrrr
#(///#
z{������{zzzzzzzzzz�������������������
#/5/+!
���������������znaXA?HUn{�MNX[gsstvww{tg[NIHIM�����������������~�������������������������#$%�����/04<IU]]a`UI<730////[hmtz���������t^XUW[Y[]hty}������th[RQUYrtu�����������ttvtr����������gt���������tpgba^^ag!#0;<?CGIKI<80+$#!/09<@=><30.,////////#-/78<?HR_UH/#EHQUWafmgaUQHEEEEEEE�)2:AGJMB5)���Pgt�����������g`XSQP���������������������������������������������������������
��#�/�>�@�<�:�/�#����ʼ����üʼּ�������������ּʼʼʼʼY�A�.�@�K�Y�f��������ʼռټʼ�����r�Y�s�g�b�a�d�e�g�s�����������������������s�H�/�"�������	��/�1�+�;�H�\�e�u�y�z�m�H�)�������)�6�@�[�t�y�y��x�t�h�O�)�Y�U�M�I�@�8�6�@�J�M�Y�f�j�r�o�f�Y�Y�Y�Y�H�;�5�-�)�%�/�;�H�a�z�����~�{�~�z�m�a�H�	�������������	��"�.�0�4�8�8�7�/�	����������z�b�C�9�N�s������������ ���ѿy�`�L�@�7�;�G�y�����ֿ����,�'����V�R�O�V�`�b�o�t�{ǆ�{�y�o�b�V�V�V�V�V�V�t�m�g�f�g�t�t�t�t�t�t�t�t�t�t�t�s�n�g�Z�V�U�Z�g�n�s���������s�s�s�s�s�s����ۿܿٿпݿ���(�A�O�Y�\�S�M�A�(�àÚÕÐÔÚàéìðù��������������ùà��ûïêì÷û�������������������������ſ;�4�0�;�G�L�T�]�`�d�`�T�M�G�;�;�;�;�;�;�m�e�f�j�f�`�[�[�`�m�y�����������������m�m�d�`�[�W�T�Z�`�m�y����������������y�m���}�{�~�����������������������������������������s�g�_�Z�T�U�Z�g�s���������������Z�M�E�A�;�5�5�7�A�M�Z�h�r�s���y�s�f�Z�)�����������������)�5�>�O�W�a�i�a�C�)�#���#�#�/�<�H�P�H�D�H�J�H�<�/�#�#�#�#�z�w�n�g�a�`�a�n�zÇÓÙÝÓÊÇ�z�z�z�z��ݾ׾ʾ������������ʾ;׾���������žŹŻ���������������$���������Ƽr�@��
����@�Y������������мӼ������r���
�����"�/�:�;�@�C�H�F�;�3�/�"��Z�:�7�A�Q�Y�g�s�����������������v�s�g�Z�������������������������������������������x�m�h�t�������������ûȻ̻̻û��������*������������*�J�O�V�]�O�C�;�6�*�z�w�m�i�m�v�z�����������������z�z�z�z�z�0�/�/�0�=�I�I�O�I�=�0�0�0�0�0�0�0�0�0�0�U�Q�H�=�<�1�<�H�R�U�Y�V�U�U�U�U�U�U�U�U�׾վվ׾�����������׾׾׾׾׾׾׾�ƎƅƁ�z�xƁƃƎƖƚƧƯƾ������ƳƧƚƎ�׾Ͼоʾ��ƾʾھ����	�����	�����ƚƏƁ�s�uƁƇƎƚƧ��������������ƵƧƚ�0�*���������������=�V�i�r�m�b�F�=�0����ľľ�����������������������������C�;�6�*�!�*�6�B�C�C�O�P�U�O�C�C�C�C�C�C�Y�V�L�@�2�/�8�@�L�r�������������~�r�^�Y�x�p�l�_�S�_�h�l�x�����������x�x�x�x�x�x�5�3�.�5�A�N�Z�g�s���������s�p�g�Z�N�A�5����������������������������������������E*E(E*E,E7E=ECEOEPESEXEPEPECE>E7E*E*E*E*�@�;�6�>�@�M�W�X�M�D�@�@�@�@�@�@�@�@�@�@�f�`�v�����ҽ���$�!����㼽������f���������¿ĿɿͿѿտݿ��������ѿĿ������������Ŀѿҿӿ׿ݿ�ݿѿĿ����������b�_�U�P�J�U�b�e�n�{ŇŔśŘŚŔŇ�{�n�bŭŨŠŞŒœŝŠŭŹ����������������Źŭ�I�G�B�I�L�U�b�m�l�g�b�U�I�I�I�I�I�I�I�I�������������(�4�A�B�A�4�(����_�^�S�O�J�H�S�_�l�x�����}�x�l�`�_�_�_�_�t�j�t�y²¿����������¿¦�������������������ùϹ۹ݹ��ܹϹù���������������������$�)�0�,�(����!�������.�G�`�l��������y�g�M�:�!�/�"��#�)�/�<�H�U�g�h�k�n�x�n�a�U�H�<�/�������������Ľнսݽ��ݽֽнĽý�����������������������������E�E�E�E�E�E�E�E�E�E�E�FFFF%FF	E�E�Eپ�����������	������	�������������O�D�A�H�Q�[�h�tāčĚěěĘđą�t�h�[�OĚĘĖėęĞħĳĿ��������������ĿĳĦĚ����������!�/�:�>�E�?�:�1�-�!�����������������ɺɺֺ�ֺɺ������������� - : N : ] 5 p - < o b > > a 0 ? S Q ` ) V d A A 2 ? � l ^ 9 V # - H _ 9 i F D \ i U @ T ' B Q g s G  Z p 7 & 0 6 % S 3 > Q - k q O 2 W d F m    n  �  B  M  �  �  �  �  +  k  �  �  B  q  ^  �  �  �  B  �  �  ?  H  M  �  �  �  �    �  2  k  �    �  D  r  o  b  �  �    �  `  n  �  �  �  �  D  �    \  �  �  �  �    �  �  S  �  �    #  y  �  M  �  �  ��T���o�P�`�0 Ž�w�T���49X�+��h���
�q����9X��9X����m�h�H�9��P���
��`B������h�@���1������/�,1�����,1�,1�#�
���T�T���'#�
�49X�'@��y�#�m�h��C���O߽H�9��-��O߽�����\)�y�#��"ѽ��w������t������C���9X������ͽ��ͽ��
���`�������Q�z����0 Ž��ٽ�h���B�rB&��B�"B�%B�Bh�B$��B�)B0�dB&b�B+3�B)BB
BĚB #B� B�B5�BY�BQB��B�qB��BM�B3IB��B)^B�BBU�A�ʼB��B�0B��B��B�B�]B ̞B�BPiB	7�B	q�B
/�A��B
1�B"��B )�B	ׁB
E�BqUB)J�B-8�B�JB%B	�BM�B,sB��B&�0B	��B��B
��B��B	�|B%�NB&�B�B�MBj�B
uxB��B<�B��B&�B�B�1B?�BA"B$9�B�
B0J�B&FtB*�TBA!B4SB��A���B?�B��BB��B}�B��B�B��BClB>~B��B2�B��B�VA�o+BF(BɭB�B�B%B�]B �^BP�B?%B	HB	?.B
CA��B
3�B"B B D�B	�]B
80BE[B)A�B-?�B�gBE>B�B$7B?�B>�B&M�B
%bB�)B9�B}�B
B%��B%�iB��B�B?�B
/�B?�BA�A�>�A��@�kA�)A�QA؉�@�BuA�W�A\��A��[Au7sB}9A���A���A��SA�l�A�]�AeMAn/�Al&FA��"A�eA?I�A��LARA���AP�A�xj@���A��A��|A�o@��A��
A�;+B
��A�r�AUޚB\AY$�B��B
�5A�TB �%?�	�@��"A���A���C���@�!�A>Az��Ax��A�-�A��mA��9A3�^@���A���>P��A�}EA�TAĠ3A&�$A2�C��MAZI_A���A��@e�`@(n�A�NA �g@��sA�A�{bA�TI@��EA��.A[�nA�@:An�B��A���A��1A�X�Â�Aπ�AcYAmbAk"�A���A��kA=�AֿRA�x�AȐBAQ��A���@�j�A�z�A��A��@�wA��EA��B
�YAĀ�AV B iAYl=BG�B
�|A�w�B ��?�Dx@���A��FA�	nC��@��A}$A} �Ay�A�q�A��A��A5��@�zA��>@�A��+A0�AÓ�A$�A3 }C��vAY-AۋgA�{�@k�g@$DM   :      2   )   %   1            F   1            ,   "                  	      ?               5            3      	            	                  %                  1                                       	      9      K            #      )   #   7               K   ?            -                           /               9                                 !      )                           3                        !         #            !                  !         #                  ;   =                                                      9                                                                                                   #                           O�]�N�.�O"zvO��O�٠O[>�N�׸Os��O�iP���Pl��Nx�cN+\�N\�4O��MOID+O�Ni��O�O6xO$��N��O���O��NT�vN�0�N}PZO8O�PLOG�CO�M�O'��OR��Or@�N��N8R�N1L�NP�O=LOMvNꗿO#�O-ԣNN�SO/�N`�O4�N�x�Nj�YN�O�ҕOY��Oy*GO8�"OLZ�N��oOe��N�/�O���N��O��O�Y>OS�9N�l�M��IO4�GN�aOV��O���O>��N��T  �  �  i  �  -  �  �  !  a    4  �  �  �  L  |  �  �    �    d  �  R  �  ]  p    �  �  O  �  	�  �  �  �  z  �  �  O  �  �  ]  �  h  U  .  �  ~  t  R    �  �  �  �  <  q  |  �  ]    �  C    �  �        `��o��o���ͻ�`B��o��t���o�D����`B��1�u�T���D���e`B�o������9X��o���
���㼣�
��1��1�T�����ͼ�j���ͼ�/��/��`B�������+�+�+�+��P��w�@��'T���<j�8Q콃o�e`B�q���u�q���q����hs�u�y�#�y�#�y�#�y�#��7L��7L������{�������T��� Ž� Ž�񪽶E���"ѽȴ9������������������������2<IU_]ZXUJII?<<62222������������������������)(����������)6CJKNN?)��#/<HSTRKH</(#��
#+#"
������:<HUanuwtonaUH<6434:$*6COX\agkh\OC6*���#Un{�����{<0���{�����������������{{��������������������#./2<8/#rtx|��������torrrrrrYamz���������zma[UUY���
#(02/*#
������QU]aez������zsna_UOQ26BMMOSOJBA<64222222[[ht{��������th[Y[Z[��������������������)6BLO[``[OB-)"�������

 ���")/3<HUaq}zaaU</*$"�����������������}|�:<HPUWUUH<<6::::::::���������� �������������������������������� ��������������)-2B6���������PTamz�������zmaYTRNPGUanz������zrnj`UQFG��������������������46:?BO[dhjiha[OB;644����������������������������� ����������&)58?65)&&&&&&&&&&��������������������
 � ��������������������EN[gt|�����tgb[ONHEEZ[]bgottxyyxtga\[YZZst����������tpkhhkssemz����������ztmgcbertw~��������trrrrrrr����������������������������������������fgt��������tqjgc`_afst��������{vttssssss
#(///#
z{������{zzzzzzzzzz�����	�������������
#/5/+!
���������������znaXA?HUn{�MNX[gsstvww{tg[NIHIM�����������������~������������������������ "$$������/04<IU]]a`UI<730////[`lt���������tg`[[X[[[htx����th[YW[[[[[[rtu�����������ttvtr����������gt���������tpgba^^ag!#0;<?CGIKI<80+$#!/09<@=><30.,////////#/468<AC>/#EHQUWafmgaUQHEEEEEEE)/8>BCA5)�Pgt�����������g`XSQP����������������������������������������������������������
��#�/�;�<�:�6�/�#��ʼ����üʼּ�������������ּʼʼʼʼY�X�Q�V�Y�f�r�������������������r�f�Y���s�g�d�c�h�s���������������������������T�H�;�/����"�/�;�H�T�a�m�l�q�m�i�a�T�6�0�)�"��!�#�)�6�B�O�S�l�o�h�`�[�T�B�6�Y�U�M�I�@�8�6�@�J�M�Y�f�j�r�o�f�Y�Y�Y�Y�;�:�1�-�-�/�;�H�T�a�p�y�w�s�t�m�a�T�H�;�	�������������	��"�.�0�4�8�8�7�/�	�����������r�V�R�W�e�s����������� ������ѿ��y�`�D�:�:�G�T�y�����ҿ����"����V�U�Q�V�b�o�{Ǆ�{�u�o�b�V�V�V�V�V�V�V�V�t�m�g�f�g�t�t�t�t�t�t�t�t�t�t�t�s�n�g�Z�V�U�Z�g�n�s���������s�s�s�s�s�s�����������(�5�A�K�P�J�E�<�5���ìäßÝØÜàìùÿ����������������ùì����ùöù�����������������������������ſ;�4�0�;�G�L�T�]�`�d�`�T�M�G�;�;�;�;�;�;�m�l�n�q�m�j�h�m�y�������������������y�m�m�f�`�\�X�U�[�`�m�y���������������}�y�m�������~�����������������������������������������s�g�_�Z�T�U�Z�g�s���������������Z�M�E�A�;�5�5�7�A�M�Z�h�r�s���y�s�f�Z����������������)�8�B�P�Q�H�B�6�)��#� ��#�(�/�<�A�A�<�;�/�#�#�#�#�#�#�#�#�z�w�n�g�a�`�a�n�zÇÓÙÝÓÊÇ�z�z�z�z��ݾ׾ʾ������������ʾ;׾���������žŹŻ���������������$���������Ƽr�@��
����@�Y������������мӼ������r���
�����"�/�:�;�@�C�H�F�;�3�/�"��Z�:�7�A�Q�Y�g�s�����������������v�s�g�Z�����������������������������������������������x�r�m�x�������������ûɻɻû������C�>�6�*�����������*�4�I�O�U�[�O�C�z�w�m�i�m�v�z�����������������z�z�z�z�z�0�/�/�0�=�I�I�O�I�=�0�0�0�0�0�0�0�0�0�0�U�Q�H�=�<�1�<�H�R�U�Y�V�U�U�U�U�U�U�U�U�׾վվ׾�����������׾׾׾׾׾׾׾�ƎƅƁ�z�xƁƃƎƖƚƧƯƾ������ƳƧƚƎ�����������	�������	������ƚƓƎƁ�vƁƌƎƚƧƳ��������ƳưƧƚƚ������$�0�=�I�V�X�a�]�V�I�=�0�$�����������������������������������������C�;�6�*�!�*�6�B�C�C�O�P�U�O�C�C�C�C�C�C�L�F�@�=�<�@�J�L�e�r�~�������~�y�r�e�Y�L�x�p�l�_�S�_�h�l�x�����������x�x�x�x�x�x�5�3�.�5�A�N�Z�g�s���������s�p�g�Z�N�A�5����������������������������������������E*E(E*E,E7E=ECEOEPESEXEPEPECE>E7E*E*E*E*�@�;�6�>�@�M�W�X�M�D�@�@�@�@�@�@�@�@�@�@�����������ּ������������ʼ��������������¿ĿɿͿѿտݿ��������ѿĿ������������Ŀѿҿӿ׿ݿ�ݿѿĿ����������b�_�U�P�J�U�b�e�n�{ŇŔśŘŚŔŇ�{�n�bŭŨŠŞŒœŝŠŭŹ����������������Źŭ�I�G�B�I�L�U�b�m�l�g�b�U�I�I�I�I�I�I�I�I����������������(�4�@�A�4�(������_�^�S�O�J�H�S�_�l�x�����}�x�l�`�_�_�_�_�u�}²¿������������¿²¦�������������ùϹѹԹ׹ҹϹù�����������������������������$�)�0�,�(����!�������.�G�`�l��������y�g�M�:�!�/�"��#�)�/�<�H�U�g�h�k�n�x�n�a�U�H�<�/�������������Ľнսݽ��ݽֽнĽý�����������������������������E�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�Eپ�����������	������	�������������O�J�D�J�T�[�h�t�~āčĖĔčĂ�t�n�h�[�OĚĘĖėęĞħĳĿ��������������ĿĳĦĚ����������!�/�:�>�E�?�:�1�-�!�����������������ɺɺֺ�ֺɺ������������� $ : B 1 6 % p " < T a A > a " E M Q \ ' T d A 9 , ? � l ^ 9 V # . G _ 9 i F D M a E . T  B Q c s G Q Z p 7 & 0 4 % Q > > Q - k q N 2 N d F m    �  �  m  �  p  �  �  �  +  �  y  �  B  q  �  �  C  �  Z  �  �  ?  H  /  h  �  �  �    �  2  k  �  �  �  D  r  o  b  f    n  r  `  r  �  �  �  �  D  �    \  �  �  �  �    _  �  S  �  �    #  �  �  �  �  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  c  y  �  �  �  �  j  I  "  �  �  q    �  L  �  >  �  �  �  �  �  �  �  �  p  ]  H  3      �  �  �  �  �  s  Y  A  (     K  �  �  �  �    T  e  i  [  2  �  �  i  .  �  �  '  {  �  �  �  �  �  �  u  o  c  Q  9    �  �  ~  (  �  *  i    �    {  �  �    &  ,  &       �  �  e     �  �    �  �    Q  �  �  �  �  �  �  �  �  �  q  A  �  �  B  �  	  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  Z  2  
  �  �  �  �        !          �  �  �  �  v  >  �  �  e  �   �  a  `  ]  W  P  H  ?  3  $    �  �  �  �  �  �  Z  )  �  �  �  �  �    �  �  d    �  :  �  y     �  �  d  )  �  )  	    1  4  (    �  �  �  _    �  �  �  t  ;  �  �  /  �   }  �  �  �  �  �  �  �  }  V  .    �  �  v  >    �  �  F    �  �  �  �  �  �  �  �  �  �  �    x  q  e  W  C  .  I  M  �  �  �  �  }  r  g  \  K  0    �  �  �  �  |  k  ^  Q  D  �           *  8  I  J  A  2  #    �  �    p  �    -  �  "  @  Z  m  w  |  s  R  %  �  �  e    �  �  z  	  �    d  }  �  �  �  �  �  �  �  d  B    �  �  ~  <  �  g  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	                �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  ~  d  D    �  �  i    �  �  R              �  �  �  �  �  p  P  .  
  �  �  n  *   �  d  X  K  D  A  ?  B  D  =  5  +        �  �  �  �    {  �  �  �  �  �  �  �  �  �  }  k  U  <  H  +  �  }    �    �  �  �      #  6  B  I  Q  L  :  �  �  ,  �  C  �  �  �  �  �  �  �  �  �  �  �  �  m  W  ?    �  �  �  z  K     �  ]  L  ;  &    �  �  �  �  �  �  o  \  I  3    �  �  Y   �  p  k  e  _  Z  T  N  I  C  =  4  &       �   �   �   �   �   �    �  �  �  �  �  �  �  p  T  2  �  �  j  %  �  �  �  �  �  �  y  ?  �  �  �  c  B  =  -    �  �  b  H  
  �  �    8  �  �  �  �  �  �  r  a  M  4    �  �  �  r  >    �  j   �  O  A  1    
  �  �  �  �  �  �  �    h  L  +  
  �  �  B  �  �  �  �  �  �  �  �  y  c  M  8  "    �  �  �  �  �  �  	�  	�  	�  	�  	�  	�  	�  	O  	  �  t    �  .  �  ?  �  �  �  �  �  �  �  �  �  �  �  �  |  ^  <    �  �  |  @    �  �  v  �  �  �  �    l  Y  F  .    �  �  �  �  �  }  �  x  ^  D  �  ~  o  `  M  ;  %    �  �  �  �  �  c  B    �  �  �  �  z  {  }  �  �  ~  p  _  B  "  �  �  �  �  [  1    �  �  w  �  �  �  �  �  �  �  }  v  p  h  _  V  N  E  =  6  /  (  !  �  �  ~  m  [  I  5  !        �  �  �  �  �  �  �  �  |  �        �  )  H  O  L  F  8  &    �  �  U  
  �  k  <  `  x  �  t  `  J  0    �  �  �  �  R    �  �  D  �  �  1  r  p  i  ]  R  k  �  �  �  �  �  g  @    �  �  U  �  F  �  Z  W  [  O  :  #    �  �  �  f  .  �  �  .  �  7  �  )    �  �  �  �  y  k  Z  I  8  (    �  �  �  �  x  [  =       �  �     =  R  `  g  g  b  U  <    �  �  O  �  i  �  �  �  U  <  &    �  �  �  �  �  �  �  v  c  Q  ?  +    �  �  �  .  $        �  �  �  �  x  H    �    :    �  �  2  s  �  �  �  �  �  �  �  q  T  7    �  �  �  �  �  �  �  �  r  ~  ^  ;    �  �  |  ?    �  �  Z  '  �  �  s  /  �  �  X  t  o  j  e  `  [  W  R  M  H  C  >  9  4  /  *  %         @    �  �  �  Q  F  8  0    �  �  p  $  �  n  �  0  -  m        �  �  �  �  �  v  O  %  �  �  �  S  �  �  C  �  v  �  �  �  �  �  |  k  W  B  )    �  �  �  j  Z  Z  w  �  �  �  �  �  �  �  ~  r  g  R  <  &    �  �  �  �  |  7    �  �  �  �  �  z  k  X  C  -        �  �  �  s  L  $  �  �  �    y  s  i  _  R  D  6  &      �  �  �  }  R  %   �   �  7  ;  1  )  #           �  �  �  �  w  Y  C  0  �  �  !  q  o  l  j  g  c  _  Y  R  K  F  B  >  :  7  4  8  b  �  �  l  x  |  x  p  c  Q  =  &  
  �  �  �  f  2  �  �  z    �  �  �  �  �  �  �  �  �  �  �  �  �  w  6  �  C  �  �  [   �  ]  J  6  #      �  �  �  �  �  �  �  |  o  c  c  f  i  l          �  �  �  �  y  T  +    �  �  {  B  �  �  w  ~  �  �  �  �  u  g  T  =  "    �  �    C  �  �  Z  �  W  �  C  :  0  *  '      �  �  �  �  �  j  d  ^  D  #    �  �      #  2  A  J  @  5  +  !      �  �  �  �  �  �  �  �  E  �  �  �  �  �  �  �  �  z  &  
�  
L  	�  	  �  �    ~  8  �  �  �  �  �  �  �  u  ^  =    �  �  �  u  J    �    �  m  �      �  �  �  b    �    �     F  q  
�  �  V  �        �  �  �  �  o  9  �  �  n  ,  �  �  c    �  V  �  R        �  �  �  �  �  �  �  �  `  2  �  �  �  =  �  �  �  `  W  M  B  5  '      �  �  �  �  �  �  e  K  0      �