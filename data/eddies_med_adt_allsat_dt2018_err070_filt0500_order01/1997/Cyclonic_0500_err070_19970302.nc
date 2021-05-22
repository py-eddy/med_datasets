CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�bM���     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�vt   max       P�B�     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�1     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F7
=p��     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v{�z�H     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q`           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @��         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       <���     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/�D     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�i�   max       B0�     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C���     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�Z   max       C��g     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          @     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�vt   max       P��     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�c�e��O   max       ?�1���-�     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�1     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F/\(�     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @v{�z�H     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q`           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @��         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CK   max         CK     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?�1���-�        `�                        (      ;            	            	      	   (         (         
   9   '   @   
             7      3   "      -         9   	               	      
                        1               
                              M�vtN��N$� N���O�B�O1n�N��OO�:�N|�	P�B�N]��N"��N6��N��EN�+YNO��P �WN��!N���OԨO�آOAO�rO�o<O!KN��O�AP%�O�NjO�%�N�L�O�[�O���O>�PO��N�x]O���O��PN���O)��N}��O)KO߁fN�r~OKUN���N��UO�N� ?O3NϚKN�oO��\N���O��NGbuN�3�O�PJ{�O),�O���O~�2N#lbOQO�N�z#O�~�NA��O6��O5N�ZTOg�N��,N
K�N�<�1<��
<�C�<#�
;ě�;ě�;��
;o��o��o�o�D����o���
���
�ě��o�o�#�
�D����o���
���
��1��j�ě��ě����ͼ���������/��`B��`B���o�+�+�+�+�+�C��\)�\)�\)�t��t���P����w��w�',1�0 Ž0 Ž49X�<j�<j�H�9�L�ͽP�`�aG��aG��aG��m�h�u�u�u��%��o��o��o��\)��������������������������FHLUamlfaUUTHHFFFFFF-/<HKHE</,----------#)6BCB@663)�����

����������bht����������tkfhkebzz������������znzzzzBHUanz�������znUJC@B&)5ABKB@5)'!&&&&&&&&���#0n������bI<����66>BOQPOLFB<6566666626ABCOOUOB9622222222IN[^gsga[SNMIIIIIIII�������������������������������������������������������������������	���������NORW[htwzxth[ONNNNNNY[\`hmtzwth^[[TRUVYY����������������������
#/9?BC=;/#����������������
#%+##
��������
  ����������7<HUalmnqnja]UH@<627"/3<HNOJHA<5/+#��������������������t|��������������wsrtTaz�������zwmaTNMPT�����������������}}������������������������������������������#*6COUY[WOE6���������������������������)*!��������&)+5=BN[afa[PNB85)&&}������������������}����! 
�������������

�����������
#%.11/,(#

�������������������������������vnnoptx���&))/50)������')57=<85)!'''''''''���!()+%�������������������������5<=HU[_\UH<:55555555�����
	

���������TUUY]_agkmquwsmlaXTT������������������������������������������������������������_fnz���������zyneVV_yz��������~zwvyyyyyyqtv�������������ytrq}����������~}}}}}}}}��������������������ABEO[bhgfa_[OKJJICBA�����"*+&���������������������������fhopot~����������tgf��������������������&)46>BCB64)+0<FIMNLKI<80,# ��������������������#0IUeghf``U<0-)-!"#*0:<GIJI@<90//******LNT[gtx����tg[NJFCDL�������������������#/49:;/,#!9<HUanwz|{ztnaUKEA>9!#/6<HHE<5/,##mt|������~trmmmmmmmm�������������������ƽݽܽܽݽ���������ݽݽݽݽݽݽݽݽݽ���������(�.�5�5�5�)�(���������������������������������������������Ŀ��������������Ŀѿտݿ��ݿѿĿĿĿľ4�.������A�Z�l�w�w���������s�M�4�û»λ׻ػܻ������ �������ܻл����������������������������������������Ҿ(������#�(�4�A�Z�i�m�n�f�\�M�A�4�(�a�_�Z�a�c�m�w�z���{�z�m�a�a�a�a�a�a�a�a�	�����s�X�O�3�.�6�N�s�������	�����	�S�R�F�>�>�F�S�W�_�l�s�l�c�_�S�S�S�S�S�S�a�]�U�T�U�`�a�n�u�u�n�f�a�a�a�a�a�a�a�a����������������������������������z�r�n�e�n�zÇÎÓÛÓÇ�z�z�z�z�z�z�z�z�n�g�a�U�Q�U�V�a�i�n�n�u�z�}Ã�z�n�n�n�n�
�	���
��#�/�3�3�/�#��
�
�
�
�
�
�
�
��ƺƳƬƳ����� ��0�6�=�@�@�:�0�$����!������ ����!�"�$�+�%�!�!�!�!�!�!�L�I�L�Y�]�e�n�r�~�����������~�r�e�Y�L�L�ʾ¾����������������þʾ׾�����׾��s�g�`�\�]�e�s�����������������������{�sŠŕŚŠŨŭŸŹ������������������ŹŭŠ����ùóììåêìù�������������������ž�ϾɾɾϾݾ��������)�.�;�8�.�"��������������	���!�"�#����	�����6�.�)�(�)�,�0�6�B�O�R�[�a�[�S�O�H�B�9�6��������(�4�:�M�P�O�M�A�;�4�(�������ֺɺ��ú���:�S�j�p�_�P�3�!����ƾƪƊƁ�r�uƅƐƧ������������������ƾ������)�B�O�[�h�w�}�}�z�t�h�[�6�)�����������*�2�6�C�O�[�O�M�C�6��y�m�c�W�O�K�I�T�X�`�������������������y�G�-��	���������	��"�.�5�;�@�F�O�M�G����������ݿѿĿ������������Ŀѿݿ���r�f�}���������.�@�C�:����ʼ������r�	�������������	��	���	�
��	�	�`�T�K�F�J�Z�`�m�y�����������������y�m�`���������������Ŀѿݿ���ܿѿĿ�������D�D�D�D�D�D�D�EEEEEED�D�D�D�D�D�D�EEEEE*ECEPEVE\EiErEiEcE\EPEKECE7E*E�������������������������������y�������������ĽͽннĽ������������z�y�6�)� ��)�0�B�OāčĒđďċĊĆā�h�[�6������������������������������������ ������� �����)�.�3�5�:�0�)����O�J�C�<�9�C�O�\�h�q�h�c�\�R�O�O�O�O�O�O���������������������������������������亰���������ɺ���!�)�3�'�$����̺������������s�g�\�g�s�v���������������������x�o�l�g�g�j�l�x�~���������������������xƧơƥƧƳƼ����������������������ƳƧƧ�@�5�4�2�+�+�'�%�'�4�@�I�M�R�X�W�[�Y�M�@�	����������"�/�;�T�a�e�s�d�a�[�T�;�/�	���{������������������������������������ŹůŭŢŨŭųŹ����������������������Ź�m�g�g�m�z�~���������~�z�m�m�m�m�m�m�m�m��������������������������������������������������������������������������������!����$��!�.�S�������Ľ������a�H�.�!��
�	�������	�
��"�/�;�H�I�I�H�;�/�"�����²¦�t�y¦²���������������b�V�I�:�0�1�3�0�(�0�=�V�b�o�z�}�}�{�o�b�a�a�T�N�H�F�H�L�T�U�]�a�j�a�a�a�a�a�a�a�����������Ľݽ��������ݽݽнĽ��������������������
��#�/�'�#���
�������������ûл�����"�$�����лû����l�g�l�m�x�~���������������x�l�l�l�l�l�l������ �#�*�/�<�C�G�P�W�Q�H�<�/�#�����ĿĳĦĠĥĦĲľĿ������������������FE�FFFF$F1F5F1F'F$FFFFFFFFF�������������������ϹԹܹ߹�޹Ϲù�����DbDbDoDpDzD{D}D�D�D�D�D�D�D�D�D�D{DtDoDb������������������������������������������� ����������������� @ & D I 3 W g 8 . j R M V c F e 4 C T N : r I 8 ( M : @ x 0 p S d S _ �  D S L D 0 / N ' K * X h @ a \ = @ 9 S ' n U 0 J - p d Y B R < B P / � f =    �  >  �    �  �  �  �  �  j  U  r  �  �  |  �  �  �  %  W  O  3  
  a  #  =  *  O  �  U  [  �  �  M    Z  S  �  �  �  o  	  �  �  �  �  G  X  Y    O  �  �  )  q  �  w  �  x  �    }  �  )  �  y  �  m  �  �    U  ?<���<u<e`B;�o����#�
�D������`B�u�D���e`B�ě��e`B��t��49X��w��o��t���1�e`B�o��/�u��w��w�C�������E��t��@��<j��%��-�#�
���罉7L�8Q콟�w�#�
�L�ͽ�j�0 Ž}�'@���o�@��}�L�ͽaG�����P�`�aG��P�`�P�`�y�#������7L���㽣�
�q����C���7L�����}󶽣�
�������^5��-���
�oB*�B�B��Ba�B��B��B�rB�5B9B%��BWB>�B��B�B!��B�bB�?B�gB�B � B��B�BսBM@B��B`�B�B.�A�h�B^LBK]B+�B/�DB�0B-?�B�B��B�ZB�B�6B
�B�wB��B�BQ�B��BM�B"�AA���B �BB�#B"E�B��BׂB
�^B
�B��B
�B�.BH�B
��B��Bv�B&�B�B&wHB&�B	WBB�B0�B.B
nB��B)��B��B�qB;�BG�B�B��B?B8,B&�)BA�B?�B҆B �B!��B�ZB��Bb�B�{B!8�B��B@�BBI�B�B��B�B�A�&B@�B?B*̄B0�B�.B.?�B�}BIBB"�B�gBD�B>�B�QB|�B*�B8�BйBLB#>hA�i�B � B�uB"AsB�?B�B
�BB
�UB��BAAB��BJ�B
?�B�OB?B&>RB�B&��B&<�B�.BG�B-�B2�B�+B
Q�B�A,��A���A��AyʼA<1@��fA�JA:��A�uVA�u�@�Q�A��A0L.A�A�;A���Bd�@eƇ?���AP��A�M�A��A�A�AXE3AY��A�AzA7zz@b��B<qA�!A��^Am)�A^�RA|�GA��AY1NAkY�Aw�=C�I�C��AӼ�A"M�A�0A�1[A��yBh:A���@DX�A���@��B�*@��gA�^wA�4�A�9�A�c�A���A��qA_�A��A���B��A��,A+AmA�-�@�9�@���A���A�1C���=��C��sA�.@���A,�aA�}�A��kAzR�A;P�@���A��A:�A�A���@�
|AƗpA1�A�}?A��A��B	vr@d��?��ZAN��A��A��A�x�AW0NAY�A؄�A7�6@c)}B�+A�\�A�S`Am�A\��A}�A�AX��Ak�Ax�C�H
C��AA��A!2Aہ�A��A�_#BA�A�}�@C�NA�G�@��B�U@��&A�}�A��VA���A�{�A� gA�D�Am�A��;A��2B�tA�r]A)3A��p@��@��A�wVAↄC��g>�ZC��A��@���                     	   )      <            
             	   	   
   )         (            :   (   @   
         !   8      3   #      .         :   	               	      
         	               2                                                            #               M                     )                     %            -   %            !      9                                       )               !                  7      !               %                                                      M                     '                                 %                     5                                       !                                 3                     %                        M�vtN��N$� N���O}�}O1n�N��OOw�4N|�	P��N]��N"��N6��N��EN��7NO��P
�#N��!N��OԨO4OOAO�rO��uN�gsN��O�AO�m�OWl>O�f>N�L�O�l5Oz�*N�!]PC�Nl�O x�Oa�rN���N��^N}��O��O0RN�r~O'��N���N��UO��N� ?N��NϚKN�oOc�ON��"O��NGbuN�3�O�PEwwO 9O|;BOS?N#lbOQO�N�z#O�~�NA��O6��O5N�ZTOG�,N��,N
K�N�  �  �  J    V  ~  w  �  �  )  �    �  �  �  ^  A  a  �  �  E     �    L    /  �  b  	�  �  �  S  c  �  �  �  �  L    ~  @  :  �  ~  .  �  u  \  S    �  �  �  Z  �    �    �  �  c  *  @  �  �    �  �      J  �  +<�1<��
<�C�<#�
��o;ě�;��
��`B��o�o�o�D����o���
�ě��ě��49X�o�49X�D����/���
���
�������ě��ě��\)�C��'�/��h���#�
�C��C��aG����+�0 ŽC���P�u�\)��w�t���P�,1��w�'',1�T���49X�49X�<j�<j�H�9�P�`�Y��q���m�h�aG��m�h�u�u�u��%��o��o��+��\)��������������������������FHLUamlfaUUTHHFFFFFF-/<HKHE</,----------#)6BCB@663)�������	
�������bht����������tkfhkebzz������������znzzzzEHJUanz������znaUOGE&)5ABKB@5)'!&&&&&&&&���#0n������{bG<���66>BOQPOLFB<6566666626ABCOOUOB9622222222IN[^gsga[SNMIIIIIIII����������������������������������������������������������������������������NORW[htwzxth[ONNNNNNY[]hltxuth[USWYYYYYY��������������������
#/29<<:7/#
��������������
#%+##
��������	
����������7<>HUaaiheaUHE<:7777"/3<HNOJHA<5/+#��������������������x���������������~xvxRWajmz~��~{wmaUQPR������������������������������������������������������������ %*6CMRUUPC6* ���������������������������() �������25ABN[_e_[TNB5222222�������������������������
�������������

�����������


#),..'#




��������������������rt������������zrpqr��$'&�����')57=<85)!'''''''''���$$�������������������������5<=HU[_\UH<:55555555���������������TUUY]_agkmquwsmlaXTT������������������������������������������������������������hnz���������znkecbdhxz��������zxxxxxxxxxqtv�������������ytrq}����������~}}}}}}}}��������������������ABEO[bhgfa_[OKJJICBA����")*%����������������������������t�����������tqmnqrqt��������������������&)46>BCB64)+0<FIMNLKI<80,# ��������������������#0IUeghf``U<0-)-!"#*0:<GIJI@<90//******LNT[gtx����tg[NJFCDL�������������������#/49:;/,#!@HJUanuy{zzpnaULHFB@!#/6<HHE<5/,##mt|������~trmmmmmmmm�������������������ƽݽܽܽݽ���������ݽݽݽݽݽݽݽݽݽ���������(�.�5�5�5�)�(���������������������������������������������Ŀ��������������Ŀѿտݿ��ݿѿĿĿĿľM�A�(�&�!�#��(�4�A�Q�Z�f�m�m�q�n�f�Z�M�û»λ׻ػܻ������ �������ܻл����������������������������������������Ҿ4�*�(����!�(�4�A�M�Z�b�e�`�Z�V�M�A�4�a�_�Z�a�c�m�w�z���{�z�m�a�a�a�a�a�a�a�a�	�����p�Y�P�4�/�7�Z�s���������	����	�S�R�F�>�>�F�S�W�_�l�s�l�c�_�S�S�S�S�S�S�a�]�U�T�U�`�a�n�u�u�n�f�a�a�a�a�a�a�a�a����������������������������������z�r�n�e�n�zÇÎÓÛÓÇ�z�z�z�z�z�z�z�z�n�k�a�U�R�U�W�a�n�t�z�|À�z�n�n�n�n�n�n�
�	���
��#�/�3�3�/�#��
�
�
�
�
�
�
�
�������Ƹ���������$�2�;�>�>�8�0�$���!������ ����!�"�$�+�%�!�!�!�!�!�!�Y�R�Y�_�e�o�r�~������~�r�e�Y�Y�Y�Y�Y�Y�ʾ¾����������������þʾ׾�����׾��s�l�g�e�g�o�s�������������������������sŠŕŚŠŨŭŸŹ������������������ŹŭŠ����ùóììåêìù�������������������ž׾ϾоԾܾ���� �	���"�"���	���׾����������	�������	���������6�.�)�(�)�,�0�6�B�O�R�[�a�[�S�O�H�B�9�6��������(�4�:�M�P�O�M�A�;�4�(�������ں׺�����-�F�O�W�S�E�9�!��������ưƧƢƘƚƧƳ�����������������������)��"�$�1�6�B�O�[�h�l�t�u�s�l�h�[�B�6�)����������*�2�6�C�O�[�O�M�C�6��y�m�d�X�P�L�L�T�`�m�������������������y�;�4�"�	���������	��"�.�8�=�@�C�G�D�;�ѿƿĿ��ÿĿѿӿݿ�����������ݿӿѿѼ�����w���������.�?�B�:����ּʼ����������������	�
�	�����������������m�b�`�V�T�S�T�Z�`�m�y������������y�m�m���������������Ŀѿݿ���ݿؿѿĿ�����D�D�D�D�D�D�D�EEEEEED�D�D�D�D�D�D�E7E,E*EE"E*E7ECEPE\EfE^E\EPECE:E7E7E7E7�������������������������������������������������������ĽȽ̽Ľ��������O�M�C�B�<�B�K�O�[�h�tāĂĂĀ�w�t�h�[�O�����������������������������������������������'�)�/�5�7�5�-�)����O�J�C�<�9�C�O�\�h�q�h�c�\�R�O�O�O�O�O�O���������������������������������������亻�����������ɺֺ��� �������ֺ������������s�g�\�g�s�v���������������������x�p�l�h�h�l�l�x�y���������������������xƧơƥƧƳƼ����������������������ƳƧƧ�@�5�4�2�+�+�'�%�'�4�@�I�M�R�X�W�[�Y�M�@�	�����	��/�;�H�I�T�W�S�L�H�;�/�"��	����������������������������������������ŹůŭŢŨŭųŹ����������������������Ź�m�g�g�m�z�~���������~�z�m�m�m�m�m�m�m�m��������������������������������������������������������������������������������!���%��!�.�S�������½������y�`�G�.�!������"�/�;�H�H�>�;�/�"�������{�{¦²¹������������¿²¦�I�?�=�4�4�=�?�I�V�b�c�o�p�w�{�z�o�b�V�I�a�a�T�N�H�F�H�L�T�U�]�a�j�a�a�a�a�a�a�a�����������Ľݽ��������ݽݽнĽ��������������������
��#�/�'�#���
�������������ûл�����"�$�����лû����l�g�l�m�x�~���������������x�l�l�l�l�l�l������ �#�*�/�<�C�G�P�W�Q�H�<�/�#�����ĿĳĦĠĥĦĲľĿ������������������FE�FFFF$F1F5F1F'F$FFFFFFFFF�������������������ùϹܹݹ�ݹϹȹù���DbDbDoDpDzD{D}D�D�D�D�D�D�D�D�D�D{DtDoDb������������������������������������������� ����������������� @ & D I  W g 9 . f R M V c F e - C F N . r I ! ' M : / [ # p M R A Y x  > S D D $  N " K * Y h : a \ 8 8 9 S ' n W # = # p d Y B R < B P  � f =    �  >  �  �  �  �  �  �  �  j  U  r  �  �  |  p  �  �  %  }  O  3      #  =  =  �    U  &  )  �  %  �    �  �  �  �    t  �  k  �  �  �  X  %    O  �  �  )  q  �  w  �  $    �  }  �  )  �  y  �  m  �  �    U  ?  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  c  M  6      J  D  >  8  2  ,  &               �                           �  �  �  �  �  �  �  x  a  J  2     �   �  �  �    >  P  U  T  O  E  3  "      �  �  �  t  )  �  �  ~  o  Z  M  A  .    �  �  �  �  �  �  �  �  x  P  N  �  �  w  t  q  k  c  Z  O  E  :  0  #      �  �  �  �  �  d  G  f  �  �  �  �  �  �  {  `  =    �  �  [  �  �    �  +  <  �  �  �  �  �  �  �  �  s  c  S  B  1       �  �  �  �  �    $    �  �  �  G  �  �  L  �  �  T  G    �  �  '  �   �  �  �  �  �  �  �  �  �  c  9    �  �  �  Q    �  �  �  I    #  %  !              �  �  �  �  �  �    R  !  �  �  �  �  �  �  �  �  �  �  �  �  x  o  g  _  V  N  E  =  5  �  �  �  �  �  z  o  b  U  G  9  *      �  �  �  �  �    �  �  �  �  �  �  �  �  �  �    h  T  D  -    �  �  �  �  ^  U  L  C  9  *      �  �  �  �  i  H  &    �  �  �  s    0  ?  2      �  �  �  q  4  �  �  N  �  �  3  �  �  �  a  [  U  N  F  =  1  $      �  �  �  �  �  �  t  o  �  �  �  �  �  �  �  �  �  x  e  P  7    �  �  �  �  O    �  F  �  �  �  v  k  _  T  I  @  ;  <  6  +        �  �  �  �  �  �  �    8  D  B  8  )    �  �  �  q    �  ,  �  �           �  �  �  �  �  |  V  /    �  �  �  Z    �  ~  &  �  �  �  �  �  �  �  �  n  Z  H  8  8  9  /      �  �  �  V  �  �  �       �  �  �  �  X     �  �  v  3  �  q     p  2  ?  F  J  K  I  F  C  9  &  	  �  �  �  n  ;    �  �        �  �  �  �  s  G    �  �  m  )  �  �  o  $  �  �  <  /      �  �  �  �  �  �  u  Z  >    �  �  �  �  s  O  *  z  �  �  �  �  �  �  �  n  =    �  �  x  !  �  9  �  �  �  k  #  O  Z  b  Y  H  7  8  ,     �  V  �  �  *  �  A  �    �  �  	R  	�  	�  	�  	�  	�  	O  	  �  \  �  �  �  C  �  �      �  �  �  �  �  �  �  �  �  �  h  6  �  �  �  x  S  /    �  t  �  |  v  p  l  \  H  0    �  �  �  �  x  G    �  �  s  -  1  O  Q  M  F  <  /  !       �  �  �  �  V  !  �  �  ^  C  7  9  <  @  [  ^  R  C  +    �  �  `    �    [  �  �  �  �  �  �  x  N  ,    �  �  �  =  �  z  +     �  �    .  �  �  �  �  �      
  �  �  �  �  �  �  c  <    �  �  �  �  �    ;  U  m  {  �  �  �  p  K    �  ]  �  g  �  �  8  �  �  �  �  �  �  �  x  C    �  �  :  �  �  E  �  I  �  �  L  7  #    �  �  �  �  b  (  �  �  U  �  �  :  �    g   �  
  
�  
�        
�  
�  
�  
u  
4  	�  	�  	T  �  �  4  �    �  ~  w  q  j  ^  Q  D  1      �  �  �  �  �  �  �  �  �  �  #  9  @  ?  <  6  .  !    �  �  �  �  �  r  Z  M  C  9  .    L  t  �  �  �    /  :  2    �  �  �  6  �  E  �  �  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  n  e  C  �  �  o  z  }  {  v  i  W  B  +    �  �  �  q  9  �  �  g    �  .  .  .  .  ,  %              
  �  �  �  �  �  k  L  �  �  �  �  �  �  �  |  l  X  @  %    �  �  �  T    �  �  ]  c  o  t  d  K  -    �  �  �  �  l  9  �  �  D  �     \  \  E  .    �  �  �  �  �  �  �  w  c  H  -    �  �    K  %  B  P  D  2    �  �  �  x  I    �  �  D  �  t  �  �   �    |  z  o  d  T  D  -    �  �  �  �  �  �  e  =    �  �  �  �  �  �  �  �  q  [  B  '    �  �  �  @  �  �  �  X  2  W  y  �  �  �  �  �  �  �  �  �  k  J    �  �  .  �  u  �  �  �  �  �  �  �  �  �  �  �  �  u  a  I  1    �  �  �  u  Z  X  U  Q  K  C  5  &      �  �  �  �  u  W  >  .      �  �  �  �  �  �  �  v  n  k  h  d  B  
  �  �  s  M  '          �  �  �  �  �  �  �  �  �  �  �  }  q  c  S  D  5  �  �  �  �  x  e  P  8    �  �  �  �  �    a  B    �  \  �  �  �  �  �  t  F    �  �  q  /  �  �  B  "  �  {  �  }  *  r  �  �  �    u  d  L  /    �  �  �  ~  5  �  �  [    h  �  �  �  �  �  �  k  H    �  �  e  /  �  �  a    *  �    8  b  a  Y  G  0    �  �  �  g  .  �  �  a    �    �  *    
  �  �  �  �  �  �  �  �  �  |  b  H  ,     �   �   �  @  4  '      �  �  �  �  �  �  v  _  F  /    �  �  B   �  �  �  �  �  �  �  �  r  \  H  5     �  �  �  �  �  �  �  u  �  �  �  �  �  �  h  F     �  �  �  �  �  �  �  �  }  U                               �   �   �   �   �   �   �  �  �  �  v  U  3    �  �  �  �  �  l  J    �  �  5  �  �  �  �  �  �  o  \  B  '  
  �  �  �  �  y  _  :  �  �     �        �  �  �  �  �  [  4    �  �  �  z  V  0    �  S  �  �     �  �  �  �  n  K  (    �  �  }  8  �  �  0  �  #  J      �  �  �  |  �  _    �  �  e  -    �  �  I  �  �  �  �  �  �  �  �  �  �  �    	  �  �  �  �  �  �  o  H     +  !        �  �  �  �  �  �  �  �  y  h  W  E  2    