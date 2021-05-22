CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�XbM��     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M߃�   max       P��     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �   max       <�/     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(�\   max       @F�ffffg     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v�\(�     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�Հ         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <#�
     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�?k   max       B4�f     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B4ȷ     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�k�   max       C���     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =P��   max       C��     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          U     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M߃�   max       P��     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�i�B���   max       ?�����     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �   max       <�1     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(�\   max       @F�G�z�     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v�\(�     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�Հ         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?ie+��a   max       ?���m\��     `  d\          "                           	                     8         "   $         	      .         !               +                              c         @   3            /                     9   	         6                     
   !   
         #   %PksN���O��sO/6�M���O���NA�Nu�lO��N_{�N��N�2�O@~�N��N���N`�8OP��OL�zP=�9N�wdM�^�Oy��O�9N<��N�Ns��O
@CP��N�~O*�O��N��
N�k�N"��N ��P��O��@O=(|NT�DN��.M߃�N,R�N�``N���OwL�P��Oӄ�OEw�O��PI�O@ �Nz�N�~UP��NJ	N[Y�N�e�N��INH.RN���O�S{N}�N[�KN d	P'�oN*�N	+TN9��P/bO�O�O"8�N��O�k�N��+M���N�N�S�O*��<�/<�1;�`B;�o:�o��o��`B�o�t��e`B�u�u��o��o��C���C���C����㼛�㼛�㼛�㼬1�ě��ě��ě��ě��ě��ě����ͼ��ͼ�����/��/��`B��h��h���o�o�C��C��C��\)�t��t���P���#�
�'',1�0 Ž8Q�8Q�@��@��D���D���L�ͽL�ͽY��Y��Y��aG��e`B�ixսixս�7L��C���C���C���C���\)���-���-��j����������)BN)�����agtu�����tg\\\aaaaaa8HNTZ^ahikigdaTH;248����)4/)��������������������������������������������������������������������������������
#),-.-##"
���������������������
#&/:>HTWUNH</#��������������������lmz����������zrmkiil"#%/<>><<1/(##"""""")*5BNQXVNB>5)��
### 
�������������	����������������������������������
&+)#
������46BOTXZOB<6444444444����������������������������������������am�����������ma[YVYa��������������������06@BOX[htwvtmh[ODB60�����������������������������������������6Jp��������pOC<#�������������������������������������������6OZ`hkqth[OB6)%��������������������#*0120-$#�������

�����������������������������%3O[h��������tYOB)"%5;HTamsy���zmTK@;325��������������������$���������������������#$//0/*#������������������������������������������������������������GJU`dnuy{��}{nf\TNNG����������������������������������EN[gt��������tg[VNCE)6O[houuqh[O)
������	#(%��������nz���������zmba\Z\an��������������������#/<GA<9/&#gnz������������wfWXg�����������������������������������st|�����������zttssyz}���������zyyyyyy����������������������)5>BFNB9)�����������������������������������������������������������������*//)��������������������������������������������������������������eh����������������ue}���������������xxx}~���������������~||~��������������������?N_t����������tgB<:?*0<@IJPUVUI<80/(****Xbcnopnb_YXXXXXXXXXX����������������������
#$+./4/#
��9<DHUaajnsqiaUH@<759�������s�Z�5�#�!�:�F�Z���������������������������������	����	������������������ƯƧƖƎƁ�vƁƚƳ�������
��� �������N�B�B�5�.�2�-�,�B�N�T�[�\�g�h�m�g�b�[�NF=F=F=F?FJFVFWF[FVFJF=F=F=F=F=F=F=F=F=F=�P�G�;�6�6�E�>�G�T�X�m�������������y�`�P�!���!�,�-�:�C�B�:�-�%�!�!�!�!�!�!�!�!�m�h�`�T�H�G�E�G�T�`�e�m�t�y�{�y�m�m�m�mE�E�E�E�E�E�E�FFF$F%F$FFFFE�E�E�E��������������(�2�4�(�������������ŽŹŲŭŭůŹ�����������������Ҿ��������������������þʾ;ξʾƾ�������������������������)�.�/�/�)�%������U�M�H�=�B�H�U�a�a�n�p�n�f�a�U�U�U�U�U�U�������~���������������������������������������������������
��
�	�������������àÛÓÇ�~�zÂÇÓàæôùý����ùöìà�����������(�5�A�M�F�A�9�5�(������Z�N�H�Q�O�S�Z�����������������������Z�U�Q�J�L�U�a�n�t�r�n�d�a�U�U�U�U�U�U�U�U�������ûлܻ���ܻлû����������������;�4�'��/�@�M�Y�f�r�z��������o�f�Y�M�;ĲıĿ�������
��#�,�#� ���������ĿĲ�����)�6�>�6�6�)�����������������������������������Ŀȿ˿ʿĿ¿������	����������	������������L�H�@�3�'������'�3�@�C�L�Z�[�Y�P�L���g�A����ֿѿ���(���������������������	����������	���"�*�#�"��	�	�	�	�	�	������������������������	�����	����l�a�S�K�S�_�u�x���������������������x�l����������������������������������껷���������ûлԻܻ�����ܻлû��������}�t�o�g�g�g�r�t�|�~�~£�ܹϹ��������ùϹ�����������������������������������	�"�,�-��	�����������������������������$�0�5�6�2�0�$�����z�y�o�u�z���������������z�z�z�z�z�z�z�zƁ�~ƁƂƎƚƧƳƿ����������ƳƧƣƚƎƁ�����$�$�0�1�0�/�$������������z�x�l�e�d�l�q�x���������������������������ּʼ��������ʼмּ�������4�/�*�.�4�4�A�I�M�M�N�M�D�A�4�4�4�4�4�4�:�-�%�!�!�$�-�:�F�_�l���������x�l�_�S�:�r�@�'�����#�4�@�M�f�r������������r�����������������������������������������������	���"�#�#�����	���g�^�R�N�Z�{���������������������������g�M�@�R�f����ʼ���������ʼ���Y�M�������Ŀѿҿݿ������������ݿѿĿ������������������������������������������������������������������������������������T�J�B�B�J�a�z�����������������������z�T�b�^�b�j�n�u�{ŃŃ�}�{�n�b�b�b�b�b�b�b�bŇŃŅŇōŔŠŧšŠŕŔŇŇŇŇŇŇŇŇùõìéììù������������������ùùùù��ƾƳƧƥƧƳƹ��������������������������������������������������������������ùʹϹչϹù��������������h�`�b�g�m�uāčĔĚĦĮĳĵĲĪĚč�q�h��������	���$�(�0�:�0�$��� ���������I�H�I�T�V�b�o�{�~Ǆ�{�o�b�V�I�I�I�I�I�I�z�y�m�l�m�r�z���������z�z�z�z�z�z�z�z�z���	�	���:�S�l���������������`�S�.��������Ľ˽нݽ����ݽнĽ������������������� �������������������������ƳƩƧƦƧƱƳ����������ƳƳƳƳƳƳƳƳĿĳĤĜĦĳĿ���������	��-�4�#�����ĿŠśŊņřŠŭŹ������������������ŹŭŠ��
���������*�6�C�I�M�C�A�6�*���t�t�n�t�}āĈčĚĤĚĕčā�t�t�t�t�t�t������·±±¸���������
��
���"��
�񽞽������������ýĽ˽ͽƽĽ��������������нνн۽ݽ����ݽннннннннн�ììùþüùìæàÝàìììììììììE*E)EEEEEEE*E7EBECEDEIEIECEBE7E+E*���������������������Ŀѿ׿ؿֿѿ̿Ŀ��� T # y B ) b h ^ E i q ,  A n Q O P ` 0 � 7 O E Z m P � # ; # # F n 2 A X B F ^ b p % Q L  S N R q # Q P J ] { J _ v L ` b r + - m \ / > 4 4 5 T b W = 9   	  �  *  s    �  V  �  N  �  x  �  �  x  �  �  �  �    �  k    p  S  F  �  <  	�  �  s  �  �  �  z  ?  �  &  �  ]  D  >  k  �  �    �  1  �  =  \  �  W  �  �  �  �  �  �  �  �  d  �  �  <  �  �  M  W      ]  �  z  �  3  @  *  g��o<#�
�����49X�D����1�t��T���#�
��C���h��j�C�������j��h�t��8Q콗�P�C���9X�aG��q�������o�o��\)�o�8Q�q���,1���t��+��hs�q���D���t��#�
��w��w�8Q�'Y��
=q�aG��u��
=��^5�ixս8Q�P�`��^5�T���Y��u�P�`�aG��y�#��;d�}�m�h�q����G��}�}󶽕���vɽ�vɽ������w���`�� Ž��;d��u��B�B	�>A�{=B��BkOB+kB,�+B!I<B^�B ݣB��B4�fB #3BN�B�8B�QB��B�]B�'Be�B*��B +B Y+B��B�B!�XB"��B&��A�?kB BH1B��B%8B�B2RB��A��uBZSB:YB5�B��B ��B!�'B��B'�BmB��B	�/Bw"B-�A�S�B"�vBBI�BcB)B�HB
~�B�B�BU	BR�BPBS|B~�Bk�B��B,�B��B
�B
� B��B	��B&W�B'ƴB*kB;nB�?B�B	��A��BIIBX�B+VB,�)B!CoB?�B ��B�\B4ȷA��`B?�B�iB1vBF1B��B��B@�B*ʜB�`B b(B��Bq�B!jB"��B(8�A�{�B:1BC�B��B%?TB?B?�B?�A�,BA�BմBNSB�B �dB!�UBE�B'��B�B"�B	�}B@�B,=fA��]B"��B7�B�-BĽB��BD7B
FJB�$B<�BB)B�kBP�B@1B�IB%�B��B>�B
��B8DB
�^B��B	¬B%�AB'�yB>1B�B��A��A�\�B�MA�6�C���Ai6�@t��Ah�C��{A3��A�KAM!�A�N?Aŉ'Ap��A�wYA�7�A�"6A�tA�i:@���@�kLA�&�A�M�AvUgA��l?���A�`BA�-A��Q@���A�V�@�g�A��DA��0>�-A���B�%A�H�B��B	��@�j0A ��A9�@��@܅aA�oHAZ�;A�>}@�RZA|gAI�NA�gA�`xA�ZA�j�A��rBm?Cm=�k�Aޒ�B	b�BQ A���A�0A(�"A0`�Bh�A�G�A�{gA�/�A���A��A$iA+0iA̵�C���AwYpA�~A�}�B�_A��NC��Ag�@ta Ah�=C���A3�A��yALzAӀ�AŃ�AqAA��NAˆ�A�E�A�y*A��@�8�@�HA�A�^�Av��A���?�0A�t�A�snA�a�@�dA�u @�"}A��A��b>�nA��B	8�A�~�BG B	�|@�8A ��A9�@xK�@�%A�_AZ�0A�x�@�)�A{ �AJ��A� �A���A�w}A��A�r{B�?1Q�=P��A�yCB	@#B�?A���A)A(L�A0��B|�A�	�A���B @�A݀�A�}�A"ŲA+�ÀC��'AwԺ          "                            	                     8         #   $         	   	   /         "         	      +                              c         A   3            /                     9   
         7                        !   
         #   %   5      !         !                                       1            %               U                        )   !                           '   #      %   ;            )                                 )            /            )                  '                                                      1                           U                        '                                 #         7            )                                 %            /            !               P�N���O(YfO�0M���OA��NA�Nu�lO��N_{�N�{�N���O@~�NO�+Np�zN?7OP��O(�DP3|�N3�M�^�OC�O��N<��N���Ns��O
@CP��N�~O*�O��N:N�N�k�N"��N ��O�r�O��#O=(|NT�DN��.M߃�N,R�N�``N���OwL�O�q�Oӄ�O��OC�*PA!�O@ �Nz�N�~UP��NJ	N[Y�N�e�N��INH.RN���O"��N�N[�KN d	P(N*�N	+TN9��P/bOM9O"8�N��O�t>N��+M���N�N�^5O*��  *  �  �  �  @  >  <    �  V  �    d    �  �  �  �  X  q  [  �  �    �  �  �  P    �  �  �  �  *  �  �  *  �  �  �  �    �  z  (  
�    �  	  ?  �  )  $  }  $    9  0  �  �  �  r      �  �  �  �  r  �  �  �  E  �  �  �  
�  
�<�C�<�1���
;D��:�o�o��`B�o�t��e`B��o��o��o��C���t���t���C���9X��1��9X�����h�C��ě����ͼě��ě��ě����ͼ��ͼ����o��/��`B��h�C���P�o�o�C��C��C��\)�t��t���\)���0 Ž�o�0 Ž,1�0 Ž8Q�8Q�@��@��D���D���L�ͽL�ͽ�+�aG��Y��aG��u�ixսixս�7L��C���t���C���C�������-���-��j��h�����)-/23)�����agtu�����tg\\\aaaaaa:>DHNTabcca^YTHE=;;:����),3.)��������������������������������������������������������������������������������
#),-.-##"
���������������������#'/<HSUVUIH</#��������������������lmz����������zrmkiil"#&/<=<<:/$#""""""""()-5BMNUQNBA5)((((((��
"
���������������	���������������������������������
%+(#
�������9BOOVUOB@79999999999����������������������������������������amz�����������zmgc`a��������������������:BHO[htuttkh[OGB::::�����������������������������������������6Jp��������pOC<#�������������������������������������������6OZ`hkqth[OB6)%��������������������#*0120-$#�������

�����������������������������)6O[h��������tl^O4&);HTZamtw}}ymaTE@:89;��������������������$���������������������#$//0/*#������������������������������������������������������������GJU`dnuy{��}{nf\TNNG��������������������������������������JN[bgt}������tg[ZNNJ)6=BFLOQMB6)�����"'"���������nz���������zmba\Z\an��������������������#/<GA<9/&#gnz������������wfWXg�����������������������������������st|�����������zttssyz}���������zyyyyyy��������������������!)059>;5)���������������������������������������������������������������(--%��������������������������������������������������������������eh����������������ue����������������{zz�~���������������~||~��������������������@GNbt���������gNB<=@*0<@IJPUVUI<80/(****Xbcnopnb_YXXXXXXXXXX��������������������
##+-.#
�9<DHUaajnsqiaUH@<759�s�Z�N�4�.�9�?�L�N�Z�g�s�{�������������s���������������	����	����������������������ƺƳƩƬƳ�����������������������N�D�B�5�0�4�/�0�5�B�N�R�[�[�g�l�g�`�[�NF=F=F=F?FJFVFWF[FVFJF=F=F=F=F=F=F=F=F=F=�`�T�G�?�:�;�G�T�X�`�f�m�y���������y�m�`�!���!�,�-�:�C�B�:�-�%�!�!�!�!�!�!�!�!�m�h�`�T�H�G�E�G�T�`�e�m�t�y�{�y�m�m�m�mE�E�E�E�E�E�E�FFF$F%F$FFFFE�E�E�E��������������(�2�4�(�������žŹųŭŰŹſ�����������������������ƾ����������������������ʾ˾̾ʾľ�������������������������)�.�/�/�)�%������U�Q�H�>�F�H�U�Y�a�n�a�]�U�U�U�U�U�U�U�U�������������������������������������������������������
��
�� ����������������àÛÓÇ�~�zÂÇÓàæôùý����ùöìà�����������(�5�B�C�A�5�4�(������Z�K�S�Q�U�Z�����������������������s�Z�U�N�T�U�a�n�q�p�n�a�U�U�U�U�U�U�U�U�U�U�������ûлܻ���ܻлû����������������M�D�@�4�/�4�<�@�M�Y�f�r�z�����|�h�Y�M��ľ���������������
�������������������)�6�>�6�6�)�������������������������������ĿƿʿȿĿ������������	����������	������������L�H�@�3�'������'�3�@�C�L�Z�[�Y�P�L���g�A����ֿѿ���(���������������������	����������	���"�*�#�"��	�	�	�	�	�	������������������������	�����	����l�a�S�K�S�_�u�x���������������������x�l��������������� �������������������������������ûлԻܻ�����ܻлû��������}�t�o�g�g�g�r�t�|�~�~£�ܹѹ��������ùϹܹ���������������������������������������������������������������������$�0�5�6�2�0�$�����z�y�o�u�z���������������z�z�z�z�z�z�z�zƁ�~ƁƂƎƚƧƳƿ����������ƳƧƣƚƎƁ�����$�$�0�1�0�/�$������������z�x�l�e�d�l�q�x���������������������������ּʼ��������ʼмּ�������4�/�*�.�4�4�A�I�M�M�N�M�D�A�4�4�4�4�4�4�:�-�%�!�!�$�-�:�F�_�l���������x�l�_�S�:�M�9�1�.�3�<�@�M�Y�f�r�����������r�f�M������������������������������������������������	��� �����	����g�b�`�k�s�{�������������������������s�g�M�E�T����ʼ�����������ּ���Y�M�������Ŀѿҿݿ������������ݿѿĿ������������������������������������������������������������������������������������T�J�B�B�J�a�z�����������������������z�T�b�^�b�j�n�u�{ŃŃ�}�{�n�b�b�b�b�b�b�b�bŇŃŅŇōŔŠŧšŠŕŔŇŇŇŇŇŇŇŇùõìéììù������������������ùùùù��ƾƳƧƥƧƳƹ��������������������������������������������������������������ùʹϹչϹù��������������t�q�n�q�t�yāĈčĚĦĪįĭĦģĚčā�t������$�%�0�1�0�$����������I�H�I�T�V�b�o�{�~Ǆ�{�o�b�V�I�I�I�I�I�I�z�y�m�l�m�r�z���������z�z�z�z�z�z�z�z�z�������:�S�l�������������`�S�.�!��������Ľ˽нݽ����ݽнĽ������������������� �������������������������ƳƩƧƦƧƱƳ����������ƳƳƳƳƳƳƳƳĿĳĤĜĦĳĿ���������	��-�4�#�����ĿşŔŏŊŝŠŭŹ������������������Źŭş��
���������*�6�C�I�M�C�A�6�*���t�t�n�t�}āĈčĚĤĚĕčā�t�t�t�t�t�t��������¹³´¿�����������
���
���񽞽������������ýĽ˽ͽƽĽ��������������нνн۽ݽ����ݽннннннннн�ììùþüùìæàÝàìììììììììEEEEEEE*E7ECEIEIECEAE7E*EEEEE���������������������Ŀѿ׿ؿֿѿ̿Ŀ��� W # Z ? ) P h ^ E i _ )  A Z S O O a 6 � + I E Y m P � # ; # 4 F n 2 C H B F ^ b p % Q L  S M 7 r # Q P J ] { J _ v L C _ r + . m \ / > 0 4 5 P b W = 3   �  �  �  f    �  V  �  N  �  :  �  �  N  �  p  �    �  Q  k  k  e  S    �  <  	�  �  s  �  P  �  z  ?  .  T  �  ]  D  >  k  �  �    h  1  r  �  [  �  W  �  �  �  �  �  �  �  �  i  Z  �  <  �  �  M  W    �  ]  �    �  3  @    g  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �        #  '  *  )  %  $      �  �  �  �  I    �   �  �  �  �  �  �  �  �  �  �  �  �  }  t  p  k  ^  ?  
  �  �  c  w  �  �  �  �  �  �  �  �  �  U    �  v  )  �  �  :    �  �  �  �  {  b  C     �  �  �  �  �  l  K    �  �  �  �  @  ?  ?  >  >  ;  0  &        �  �  �  �  �  �  �  �  �  �    +  6  <  <  4  &    �  �  �  �  �  r  N  !  �  �  �  <  ;  9  7  6  4  2  1  /  .  )  !           �   �   �   �              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  K  #  �  �  �  k  9    �  b  �  �    Q  V  T  R  P  N  K  I  C  ;  3  +  #         �   �   �   �   �  P  �  �  {  _  ?    �  �  �  �  f  /  �  �  �  ?  �  o  �            	  �  �  �  �  �  �  �  �    d  H  (     �  d  ]  Q  D  9  -       �  �  �  �  h  :    �  j    �  T  �      
    �  �  �  �  �  �  y  ^  A  %  	  �  �  �  �  �  �  �  �  �  �  �  v  h  Y  N  G  ?  :  :  9  6  %      �  �  �  �  y  d  J  ,    �  �  �  �  `  ;    �  �  �  y  �  �  �  �  �  �  �  o  D    �  �  �  [  &    �  |  w  D  �  �  �  �  �  �  �  �  �  g  :    �  d  	  �  E  �  ]  �  S  W  K  =  .      �  �  H  ,  :  2  4    �  v  �  
  P  '  =  R  e  s  |  �  �  |  n  R  +    �  �  ^    �  x  #  [  Q  G  =  3  )           �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  c  2  �  �  m     �  `  �  P  �  �  �  �  �  �  �  �  �  �  �  a  1  �  �  :  �  (  �  �        (  :  M  `  {  �  �  �    K  �  �  �  D  �  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  X  >  �  �  �  y  n  e  _  Y  @  $    �  �  �  �  ]  :    �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  U  <    �  �  �  �  P     �  �  M       W  [  L  :  &    �  s    �    �   �    �  �  �  �  �  �  �  m  Y  E  2  #      �  �  �  �  �  �  �  �  �  �  �  �  v  Z  7    �  �  �  P    �  h  �  T  �  z  X  :  "  �  �  �  L  �  �  z  \  j  i  W  <      L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  +  �  &  �    �  �  �  �  �  s  \  E  .    �  �  �  �  v  "  �  a  f  �  *  #      �  �  �  �  a  ?    �  �  �  �  x  U  1     �  �  �  s  `  N  :  !    �  �  �  �  r  N  *     �   �   �   s  [  x  �  �  h  E  (  �  �  �  �  q  <  ,    	  �  k  �  �  �  �      &  )  %    �  �  �  �  a    �  &  �    �  �  �  �  �  �  �  �  e  C  !  �  �  �  m  5  �  �  Y    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  l  `  T  �  �  �  �  �  �  �  u  j  _  R  D  5  $    �  �  �  �  �  �  �  �  �  �  �  b  B  "     �  �  �  w  T  0    �  �  �            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  P  >  *    �  �  �  �  �  e  5  �  �  z  _  D  )    �  �  �  �  �  �    q  b  S  E  5  %      (      �  �  �  �  �  ~  d  I  +    0  5  "    �  �  �  	K  	�  
  
V  
�  
�  
�  
�  
�  
  
B  	�  	�  	&  �  �  �  �  |  D    	  �  �  �  �  �  �  l  L  -    �  �  �  y  W  .  �  �  �  �  �  �  �  �  �  �    Z  2    �  �  U    �  �  7  �  A  L  a  n  �  �  �  	  �  �  �  L  �  w  �  _  �  m  P  <  9  >  4    �  �  �  R  
  �  f    �  *  �  @  �  �  �  �  �  �  |  h  N  9  "    �  �  �  b  ,  �  �  ^  �  x   �     )  (  '  &  %  $  #  "  !  !         �   �   �   �   �   �   �  $          �  �  �  �  �  �  �  �  �  i  Q  6    �  �  }  o  ]  P  O  G  9  $      �  �  �  �  H    �  5  �    $        �  �  �  �  �  �  �  �  �  �  �  �  t  Y  >  #                 �  �  �  �  �  �    g  N  4    �  �  9  +        �  �  �  �  �  u  \  E  /    
    h  �  x  0  &      
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  N  /            �  �  �  �  �  �  v  �  �  �  �  �  �    k  U  ?  (      �  �  �  �  �  y  k  
�  
�  T  �  �  �  {  X  %  
�  
�  
*  	�  	  Q  �  �    �  `  G  U  b  j  p  v  |  �  �  �    <  6  /  &        �  �            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  u  j  _  c  l  u  ~  �  �  �  �  �  �  v  X  0  �  �  u  4  �  �  �  5  �  D  K  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Y  �  �  �  �  �  �    u  j  \  M  ?  (    �  �  �  �  Y  1  �  �  �  �  �  {  `  A     �  �  �  �  t  O  *    �  �  �  r  W  ;  !    �  �  �  m  ?  	  �  u    �  [  7    �  �  w  �  �  �  �  �  �  y  `  @    �  �  +  �  r    �    }  �    w  o  e  X  F  1    �  �  �  �  j  @  #  	  �  �  x  �  �  �  �  ~  f  M  .    �  �  �  ^  "  �  �  O     �   k  :    C  <  3  &    �  �  �  �  M    �  N  �  T  �  %  X  �  �  �  �  t  W  :    �  �  �  �  �  �  �  {  h  [  w  �  �  �  �  �  �  �  �  �  �  �  �  y  d  I  /    �  �    L  �  �  �  �  i  :  �  �  X    �  \    �  T  �  �  <  �  y  
3  
u  
P  
-  
	  	�  	�  	�  	^  	&  �  �  S  �  �  #  �  :  �  #  
�  
�  
  
>  
  	�  	|  	"  �  s    �    �    �    o  �  �