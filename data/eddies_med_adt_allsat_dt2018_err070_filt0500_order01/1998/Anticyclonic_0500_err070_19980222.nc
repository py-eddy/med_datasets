CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�+I�^      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N	�   max       P�2      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �q��   max       >o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @E������     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vq�Q�     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�t�       max       @��@          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �aG�   max       >P�`      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ن   max       B+O�      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�c   max       B+��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =c�   max       C�n�      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =^y�   max       C�w�      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nl�   max       Pw�z      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?يڹ�Y�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �q��   max       >	7L      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?#�
=p�   max       @E������     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vp�\)     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�B�       max       @��`          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?م�oiDh     �  V�      	      L                  _            F         9   e   /      	               O   �      
   8            	               8                  J      '      ;         
               
   #   :         ;   'N��[N���Oh(�P߸N��N*�N':�N�`
N[P�2O�Q�N/�NR��Px��O
ANy�Pa�P8�^O4!�P6-N�g�N��ORh�N�ɆO>	}P�@\P�^OAe,N��"P%�1Np��OQ�N��rN��WNO�"O6�O�T�N'��P`�OYr�N��N6��N�� N��fP�NPsKO
SO3�OаnN���O��jOuNF�@N	�N�w5N�i�N�	�O��O�o�N͹�NT�nO�c�OR��q����1�e`B�t���`B��o��o��o:�o;D��;�o;��
;��
;��
;��
;ě�;ě�;�`B<o<t�<D��<T��<e`B<�t�<���<�1<�j<���<���<���<���<�/<�`B<�=o=+=+=C�=\)=\)=�P=�P=��=#�
=,1=0 �=L��=P�`=]/=]/=aG�=aG�=}�=�%=�%=�+=�\)=��P=��P=�1=���=���>o��������������������CIN[gktwthg[ZNCCCCCC����������������������������������������{t��������������{{{{��������������������BBHO[cb[OBBBBBBBBBBB63;<=HUX_XUHB<666666��������������������$/DKa��������nU</[YWXX[`hy���������h[agjt}���ytigaaaaaaaa������������������������
&<Han{woH/����qsz~������������}zqq)44)�����#/GYZWOH</����������
�����������
#$#
�������)5Q\[N5$��)26=ABHEB6)(��������������������-*)+-/7;HT`^]^UMH;/-������������~}����������������~ 6N[���������ti^N5�����5MN[l}��H5����������

��������

�������������
 /<UZ\TB5��y{������������yyyyyyJITWamz~���}zqmdaVTJ104<FHUY]ZUKH<111111�������������������������������������������������������������������������������BBO[[[hb[OIBBBBBBBBB����)B[��|�t[B)��������������������������������������������������

#.0770/-#


-++/;<HLPU[]UQH<5/--��������������������O[\ehtwxtrh[OOOOOOOO)%(,/7<HKRTTOH</))))�������

��������������� ���������������������������/;HNTX[]YTH;+""%+/��������������WN[`gotutlgf^[WWWWWW
�����
vuux{~���������{vvvv����������������������������������������)--,((������)8CHE@6)�
#,/10/#
	�������������������������������������������nmnpz~�����������ztn�������������������������������������������������������������������������������6�C�L�O�T�T�O�B�6�*����������&�6�������������	�������������x�k�h�����[�g�k�l�l�m�g�g�f�[�N�J�N�Q�U�W�[�[�[�[���������������������������������������ҽy�y���������y�t�l�x�y�y�y�y�y�y�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eپ(�0�4�A�G�K�A�;�4�(�$�&�(�(�(�(�(�(�(�(���A�c�����w�j�_�N�A�������ٿտ����(�A�M�f�v�����z�f�M�A�4�(������ʾ־׾ؾ׾վʾľ��������ʾʾʾʾʾʾʾʾ�������������۾׾�������㾾�׾�����Ӿ̾žǾ����s�����t�����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;Z�f�n�r�n�f�Z�R�T�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�/�H�T�[�f�j�f�]�;�"�	���������������	�/�������5�@�@�:�5�)������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Ź������������ŹŔ�x�o�s�{ńŇŔŠūŭŹàãàÕàäàÓÏÇ�z�x�o�w�zÂÇÓÔà���������������������������~���������������	��"�/�1�;�@�D�E�;�/�"��	�����������ûƻлܻ������ܻлŻû������ûûûû������%�!����� ����߻ڻڻܻ�����	�"�1�/�8�	�������x�_�S�R�]�s��������������)�;�F�h�t�z�z�[�)���������������'�-�4�@�S�d�Z�]�Y�S�M�@�4�'������r�������������r�l�g�n�r�r�r�r�r�r�r�r���������,�5�$������ƳƚƎ��~ƞƭƳ���������ùʹϹعϹù���������������������čĚĤĦĩĨĨĦĚďčā�}�t�p�t�vāĆč�n�zÇÌÓ×ÛÓÇ�z�u�n�l�j�n�n�n�n�n�n�Ϲܹ߹����ܹϹŹù����������ù͹Ϲ��zÀÀ�{�z�n�g�a�U�P�U�Y�a�m�n�z�z�z�z�z�Z�f�s�s�x�v�z��s�f�Z�V�A�3�.�4�=�A�M�Z�"�.�;�G�T�`�b�`�V�H�;�"��	���	���"���������������z�w�w�z�������������������m���������������������m�T�D�A�1�.�6�T�m���Ŀѿ�����
�����ݿϿĿ��������������ʾ׾�����������׾˾ʾ�����������ÇÓÜàâàÓÇ�ÀÇÇÇÇÇÇÇÇÇÇ�f�s�y�����������������t�s�f�]�\�e�f�f�������� ���	��������������������g�������������������s�g�Z�N�H�E�F�K�Z�g�:�E�F�S�W�U�S�F�=�:�2�2�:�:�:�:�:�:�:�:D�EEEE$E*E/E-E*EEED�D�D�D�D�D�D�D�����������������������������ŹŷŷŹſ���b�nÇÓÖÕÐÁ�z�n�a�U�H�8�5�3�5�D�U�b����������������������|�����������������V�\�^�I�D�<�0�#���������
��1�<�I�V��#�.�$���
���������������������
���{ǈǋǈǄ�{�u�o�g�b�b�b�o�w�{�{�{�{�{�{āĀ�t�o�h�b�h�tāąāāāāāāāāāā�нݽ�������������ݽнνƽннн�����������������������������������������²³¿����������¿²±¨¦¤¦®²²²²�!�:�F�S�V�V�S�G�:�-�!�����������!�������ɺ��ܺӺɺ����������x�v�y�~����ǮǮǲǭǩǡǙǔǈǅ�|ǄǈǔǔǠǡǮǮǮ�.�:�;�G�I�G�=�:�.�(�!��!�(�.�.�.�.�.�.�ʼּ����������ּʼ��������������ʻ������������������������x�p�l�j�s�}���� / c % L r v I K g 1 U : ? Y ' U V 0 8 O _ 8 5 G 8 j > D & W e 0 K L � 5 ; [ Q u h N I  ? g  P  m 6 � h o ] G 5 ? $ Y - F %    �  �  �  �      X  �  Y  �  h  Q  k  B  0  �  7  3  �  z    �  �    �  �  �  �  �  Y  �  J  �  �  �  �  A  _    0  �  i  �  	  �  �  0  q  �  �    �  `  Z  �  �  �     r    `  c  ��aG��D��;�o=�+�o:�o:�o;�o;ě�=ȴ9<�j;�`B<49X=���<�<49X=�%=�"�=]/<��<��
<���<���=t�=�w=���>P�`=��=\)=��T=��=49X=,1=��=�w=�o=<j=�w=�Q�=�%=P�`=D��=<j=q��=�h=H�9=�9X=��P=�`B=ix�=��
=��=��=���=��
=��P=���=�;d>+=Ƨ�=�S�>!��>)��B��B��B�NBǲBNB�AB�B<TB"2B�PBB	�_B*�B��BcB^�B�-BO�B B��BLPB"T�A��B��B�IB	G�B��B#�B#˫B�:B&�A�=�B-�B�*Bo�BN/B!�B�B-B�7B�XB!�BB$�iB�6B�iB�B�RB��B�KB+O�A�نB�0B	!�B�B)A�B�aB��B4�B�BA�B��B�B-B�<B��B�[B��BF�BɟB�CB�5B"?BA?B:�B	�8B@'B�VB=.BxaB�B@B:EB�B��B"@A�cB�B�dB
=:B��B#�bB#��B%kB;=A��B?�BkzB��B?]B!�BeB��B��B�B"AB% �B�bB��B��B�	B�B/�B+��A��B�B	@KB@B)A�BµBI�B@B��B:'B�BD
B�A�g�A���A�bdA��A�!'A���A�8C�n�A9F�A�ĠA;�AQ�AV2EAO��C�b�A@z�A��A�gFC���A���A��@�		A�-�@�z@���A��A��@�B-@�^oB\�=c�A���A��G>��A���A?LAa��A�X�Al��A|%5AS��A���AC+�A�6xA��O@��_C�b�A�1�A��A��AꡗA穒BHAܰ�A-4�A�AcA��`@o�@��B{pA�LA Ҭ@���A��A��mA��A�a�A��A��A�C�w�A9�A�m�A<�:AQ�AV�AOVC�g�A@�~A���A���C��HA��A���@�)A��;@��@��GA�{oAԩI@��@�*5B�=^y�A�|�A�}�>�ТA��A=�%A`;EA��:Ai%�A~�hAR��A��AC�A���A���@|��C�_'A�o�AƉA%�A�h�A搇B��A�cA.�A��A�3@{�b@�OB� A�K@��@��&      
      L                  `            G         :   e   /      	               P   �         8            	                8            	      K      '      ;                     	   
   $   ;         ;   '            +                  7   !         ;         7   )      )                  =   5         1                           3                  )            !                                                                        +                     !         %                  =            '                           3                                                                        N��[N���N�RO�K�N��N*�N':�N�`
N[PB'�O�;�N/�NR��O�GFN�Ny�O̙�O{vPN_|`O��^N�g�N��N�N�ɆO�Pw�zOWD�N�M�N?�RO�U�Nl�N��3NN�RN��WNO�"O&��O�T�N'��P`�OB�sN��N6��N��&N�O�:�NPsKO
SN���O��N���O:�OuNF�@N	�N�w5N^��N}�Ou�zO`��N͹�N9�O�!�OR�  �  �  �  S  �  <  �  �  v      �  �  D  )  �  �  
�  �    ,  _  �  s    	�  �  �  �  �    M  �  �  b  (  �  �     %  C  +  s  �  	�  �  
�  �  
   �  �  Z  H  R  �  �  �  �  ?  �    
�  	x�q����1��`B<����`B��o��o��o:�o<�`B;ě�;��
;��
=�P<#�
;ě�<�h=e`B=o<#�
<D��<T��<�t�<�t�<�j<���>	7L<�`B<�`B=�P<�<�h=o<�=o=\)=+=C�=\)=�P=�P=�P=�w=0 �=u=0 �=L��=Y�=��=]/=�%=aG�=}�=�%=�%=�7L=�hs=���=�-=�1=��=�
=>o��������������������CIN[gktwthg[ZNCCCCCC����������������������������������������{t��������������{{{{��������������������BBHO[cb[OBBBBBBBBBBB63;<=HUX_XUHB<666666��������������������++1<Ha���������nU</+YZZ]dht��������th_[Yagjt}���ytigaaaaaaaa��������������������������
&1970#
���yyz������������zyyyy)44)����
#/CKNMKD</
����������� ������������

������������)5PU[ZN5(��)26=ABHEB6)(��������������������.-//5;CHNTWVVTMH;/..������������������������������$:Nt���������tl`N5	)58BEGF@65)	������

����������

����������������)1FNSSPGB5�}~����������}}}}}}}}NT[amz|��{zomiaYUTNN;59<HUWWUH><;;;;;;;;�������������������������������������������������������������������������������BBO[[[hb[OIBBBBBBBBB����)B[��|�t[B)��������������������������������������������������	
##$040.+#
				/-//<HMTUZUMH<7/////��������������������O[\ehtwxtrh[OOOOOOOO)%(,/7<HKRTTOH</))))�������

���������������� ����������������������������**+/1;HPTWWTRHH;2/**��������������WN[`gotutlgf^[WWWWWW
�����
vuux{~���������{vvvv������������������������������������ ������)--,(')69=>:5*)�
#,/10/#
	�������������������������������������������nmnpz~�����������ztn��������������������������������������������������������������������������������*�6�9�C�I�C�@�6�*���	������������������������������������������������[�g�k�l�l�m�g�g�f�[�N�J�N�Q�U�W�[�[�[�[���������������������������������������ҽy�y���������y�t�l�x�y�y�y�y�y�y�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eپ(�0�4�A�G�K�A�;�4�(�$�&�(�(�(�(�(�(�(�(��(�A�W�b�i�d�Z�N�A�(��������������A�M�f�s����}�u�f�Z�M�4�(������&�A�ʾ־׾ؾ׾վʾľ��������ʾʾʾʾʾʾʾʾ�������������۾׾�������㾱���ʾ׾������׾ʾ�����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;Z�f�n�r�n�f�Z�R�T�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��"�/�;�H�P�S�N�D�/�"��	��������������������������
�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Ź������������ŹŭŔ�{�p�t�{ŇŔŠŮůŹàãàÕàäàÓÏÇ�z�x�o�w�zÂÇÓÔà���������������������������~�������������	�� �"�/�7�;�=�;�5�/�"���	�����	�	�ûƻлܻ������ܻлŻû������ûûûü����������������޻߻�������	�"�/�.�"�	���������c�Y�X�c�s������������)�4�7�7�6�.�)�����������������'�4�@�D�M�Y�S�M�@�@�4�'� �!�����'�'�������������s�r�y�����������������������������ƳƧƚƎƉƋƜƧ���������ùùƹù�������������������������ĚğĦĨħħĦĚčā�~�t�s�t�zāčĐĚĚ�n�zÇÉÑÎÇ�z�x�p�n�m�n�n�n�n�n�n�n�n�Ϲܹ߹����ܹϹŹù����������ù͹Ϲ��zÀÀ�{�z�n�g�a�U�P�U�Y�a�m�n�z�z�z�z�z�Z�f�p�s�w�u�{�s�f�Z�M�A�6�4�0�4�>�A�M�Z�"�.�;�G�T�`�b�`�V�H�;�"��	���	���"���������������z�w�w�z�������������������m���������������������m�T�D�A�1�.�6�T�m�����Ŀѿ����������ݿпĿ������������ʾ׾�����������׾˾ʾ�����������ÇÓÜàâàÓÇ�ÀÇÇÇÇÇÇÇÇÇÇ�s�v����������x�s�f�_�^�f�h�s�s�s�s���������������������������������g���������������������s�g�Z�S�O�N�P�X�g�:�E�F�S�W�U�S�F�=�:�2�2�:�:�:�:�:�:�:�:D�EEEE$E*E/E-E*EEED�D�D�D�D�D�D�D�����������������������������ŽŹŸŹ�����a�n�zÇÌÑÑË�}�n�a�U�H�@�<�:�;�H�U�a����������������������|������������������#�0�:�<�A�C�<�8�0�#���
�
��
�����#�.�$���
���������������������
���{ǈǋǈǄ�{�u�o�g�b�b�b�o�w�{�{�{�{�{�{āĀ�t�o�h�b�h�tāąāāāāāāāāāā�нݽ�������������ݽнνƽннн�����������������������������������������²¿����������¿µ²©¦²²²²²²²²�����!�-�:�F�S�V�U�S�G�:�-�!������ﺰ���ºɺϺκʺ������������~�~�}��������ǮǮǲǭǩǡǙǔǈǅ�|ǄǈǔǔǠǡǮǮǮ�.�7�:�G�H�G�<�:�.�*�!��!�*�.�.�.�.�.�.�ּ������������ּʼ��������������ʼֻ������������������������x�p�l�j�s�}���� / c % A r v I K g 0 R : ? % # U : #  R _ 8 5 G : j  4 ; Q d ) < L � 8 ; [ Q o h N = ! ' g  N  m  � h o ] I 7 9  Y + A %    �  �  �  I      X  �  Y  H  7  Q  k  >  �  �  �  �  l  Q    �      ?  �  �    [  o  M    r  �  �  y  A  _    �  �  i  �  �  �  �  0  O  D  �  '  �  `  Z  �  �  �  �  �    N    �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  z  s  l  e  _  Z  T  O  M  N  O  P  R  [  f  r  }  �  �  �  �  �  �  u  b  H  ,    �  �  s  6  �  �  p  *   �   �  p    �  �  �  �  �  �  �  �  �  �  �  g  J  '  �  �  �  �  �  M  �  �    :  L  S  L  5    �  r    �  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  R  <  F  Q  [  b  \  U  O  H  A  9  1  *  #          
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  p  b  U  G  �  }  l  \  I  /    �  �  �  �  z  Z  :     �   �   �   ~   T  v  s  o  l  f  Z  O  C  9  1  )  !  	  �  �  �  �  t  ^  I      �  �      	  �  �  d    �  9  �  x    �  1  o  �  �      �  �  �  �  �  n  T  <  '  -  O  K  =  -    �  �  �  �  �  �  �  �  �  x  p  g  \  M  ?  1  #       �   �   �  �  �  �  �  �  �  �  y  q  i  ^  Q  D  3      �  �  �  �  P  �  �    6  V    /  A  C  ,  �  �  h    �  �  	  �   P      #  (  (  #      �  �  �  �  p  :  �  �    �  %  �  �  �    {  v  r  n  i  e  `  \  W  T  S  R  Q  N  L  I  G  {  �  }  i  Z  V  �  �  �  �    O    �  C  �  E  �  %   �  �  �  N  �  	b  	�  
  
Q  
u  
�  
}  
]  

  	x  �    +    @  �  r  �  �  �  �  �  �  �  �  �  �  �  �  �  T  �  W    �  x         �  �  �  �  �  �  t  L    �  �  �  �  m  @      ,  +  *  "      �  �  �  �  t  r  l  V  @  *    �  �  �  _  Y  S  J  >  /      �  �  �  �  R    �  o    �  3   �  �  �  �  �  �  �  �  �  �  �  �  }  Y  .    �  �  �  �  �  s  \  ?  &    �  �  �  �  X  '  �  �  S  �  �  *  �  N  �  �  �  �      �  �  �  �  �  �  �  ^  8    �  �  Z  �  O  	m  	�  	�  	e  	E  	  �  �  X    �  |  <  �  �  h  �  �  (  )  u  �        7  �  �  �  
  `  �  �  ^  �    �    �  ?  �  �  �  �  �  �  �  �  �  �  x  y  z  p  `  M  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    h  Q  ;    �  *  n  �  �  �  �  �  s  D    �  �  �  W    �  <  �  �  �  A  M  X  c  �  �     5  m    �  �  >  �  �  n  �  b  �  <  3  D  L  L  D  .    �  �  �  `  -  �  �  �  V  �  ]  �   �  �  �  �  �  �  �  �  �    g  G  "  �  �  �  V    �  �  �  �  �  �  �  t  a  K  5      �  �  �  �  r  R  3    �  �  b  ^  [  X  V  U  R  O  K  E  =  5  )      (  4  /  "      &  &       �  �  v  B    �  �  �  Z    �  ^  �  �  �  �  ~  t  k  g  b  ]  V  M  B  5  #    �  �  �  �  �  c  B  �  �  |  q  e  X  K  =  -      �  �  �  �  �  }  d  J  1     �  �  �  �  �  �  �  �  �  �  T    �  �  �  w  �    �  �        �  �  �  Q    �  �  Z  F  ?  �  n  �  h  �  l  C  5  *  !      �  �  �  �  �  w  ?    �  �  O    �  �  +  .  /  -  (  "      �  �  �  �  �  �  �  �  �  p  W  ;  f  k  q  r  p  q  y  �  u  f  W  H  9  &    �  �  �  �  �  �  �  �  �  �  �  �  r  L    �  �  �  M    �  �  E    N  a  �  	6  	o  	�  	�  	�  	z  	\  	8  	  �  l  �  d  �  �     -     �  �  �  �  �  �  �  �  o  Z  F  2      �  �  �  �  �    
�  
�  
�  
Z  
+  	�  	�  	�  	R  	  �  ~    �    i  �  �  �  �  �  �  �  �  �  �  �  }  e  F  "  �  �  �  G  �  r  �  /  }  	�  	�  
  
  
  
  	�  	�  	�  	g  	(  �  Z  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  o  j  p  v  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  E    �  �  �  �  �  Z  @  '          �  �  �  �  �  q  I    �  t  �  t   �  H  <  0  $  (  :    �  �  |  >  �  �  j  $  �  �  I  �  �  R  =  '    �  �  �  �  y  W  3    �  �  �  p  I  !  �  �  �  �  t  a  ]  S  :  (  (    �  �  �  |  :  �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  a  J  (    �  2  �  �  �  �  �  �  �  �  �  �  �  �  �  x  l  Y  C  $    �  �  r  �  �  v  Y  +  �  �  u  9    �  �  =  �  W  �    _  �  
  2  :  =  >  ;  1    �  �  �  �  R    �  3  �  p  �   d  �  �  |  r  b  G  $  �  �  �  l  2  �  �  t  ,  �  �    |         �  �  �  �  �  {  Q  $  �  �  �  r  A    �  �  h  
�  
�  
�  
�  
�  
�  
s  
@  	�  	�  	L  �  �    �  �  e  �  �  8  	x  	J  	,  	  �  �  k  $  �  z    �    �    �  V  �  h  �