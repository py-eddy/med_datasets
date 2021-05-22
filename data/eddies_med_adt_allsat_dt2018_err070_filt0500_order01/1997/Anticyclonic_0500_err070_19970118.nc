CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?° ě��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NE   max       P��+       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       >C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @F�\(��     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vh�����     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q@           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�J�           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >ix�       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�t   max       B/:       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{0   max       B/?
       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�V   max       C�p       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >W�   max       C��       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NE   max       PW��       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��W���'   max       ?�_o���       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       >C�       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F�\(��     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vf�Q�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q@           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�q�           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C    max         C        Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��N;�5�   max       ?�_o���     �  Zh               
         ;      u   .      p                     (               	      ~   -   (   2   $                        >      
      +         $               	      &                  %   �                  
   GNŎ�N!�NơNk�iN qgNZ�OiqP��O:�P��+O���N��}P�!O�ȩN&��N���N���NZ1TN�f�Ou��O%�/N�l=Ou]{N�jN�MO��P�lGP��Om1P+|!O�ZN�1OO��	N�.�N��N��FO2N�N��P��O'u�Nx��N�d�O���O �NFg�OV��Ng�N�.GN��NEN&��O�O���N��WN��N��O�O��OmPQ2~N~9kN��@N�VfN���N���NG��O0fu��1��C��49X�t���`B��`B��o;�o;�`B<#�
<49X<49X<u<u<u<u<�o<�C�<�C�<�1<�9X<�j<ě�<ě�<ě�<ě�<���<���<���<�h<��=o=o=o=o=C�=��=��=#�
=,1=,1=0 �=49X=8Q�=8Q�=<j=H�9=P�`=P�`=T��=q��=y�#=}�=}�=�%=�o=���=��w=��
=���=�9X=�j=Ƨ�=���=�F=�>C�������

��������W[gt||tg_[WWWWWWWWWW��������������������hjtw�������thhhhhhhh����������������������������������������5:BDIbln{����{nbUI<5NNQ[ht����������h[TN��������������������������)8;5������xnnqz�������������zx���
#/<A></+#
������)BNWGBEN[a[ND���HBDGNU[_g�������t[NH&()6=?6-)&&&&&&&&&&*+2/*$NOQ[_ht���th[ONNNNgfimz{~~zsmgggggggg)257=654)$fht����������~rqnphf�������

����������

����������ZYY_ajryz{������zmaZ��������������������������������������� �������68-!���������#((/9=NQanrzkUH/#��~����������������1147BNg������{tgNC813834=BOgt������tgO63��������������������
	 )6BJLD>;6.)30.6BMOPOLIB>6333333��������������������87:<HU`anona^UH<8888�������������������������������������������&*'������
#)//+&#
##$$#��������������������
)5@FJLPSTNB)
�����������������������



����������	"/;?CA;9/"	������������������������������������������������������������������������UH??<///2/./<HUUUUUU��������������������������(,--$����CDHJUahntnfaUHCCCCCC	
#$)$#
	��������������������a`amuz������zmga`_aa
��������
������������������������������������xqnrwz������{zxxxxxx��������������������UWacnrz~��}znaaYUUUU���
"#'(&#
�����,(),/<HHG></,,,,,,,,������������������������������ ������ֻ:�F�S�_�d�e�_�]�S�F�:�3�6�7�:�:�:�:�:�:�����������������������������������������~�������������������������������~�~�v�~�L�Y�d�e�j�e�e�Y�O�L�H�H�L�L�L�L�L�L�L�L�����)�6�:�6�*�)�������������'�!�������������������������������üƼ��������������x�s�s��s��������ľξѾоʾ�����f�Z�F�D�H�Z�s�4�A�M�O�P�J�A�4������ݽн�����(�4���������"�Z�a�`�H�/��������������������h�tāčĦĳĸĽ��ĹĳħĚā�s�h�e�d�f�h����&�"�!�!�&���������	������<�b�vŊśœ�{�U�<�����ĿğēĦ�������O�[�h�tĂčĒā�}�n�h�a�O�B�6�'�,�@�I�O�ݿ�����������ݿܿݿݿݿݿݿݿݿݿݿݿ`�m�y�z���������y�m�f�`�]�]�`�`�`�`�`�`�-�.�:�C�F�O�P�F�C�:�/�-�'�$�&�,�-�-�-�-���������������������������������������˿.�;�G�N�T�[�T�K�G�;�.�.�.�5�.�"�.�.�.�.�U�Y�_�`�]�V�<�/�#������#�/�<�H�Q�U�f�s�����z�z�}�s�o�f�Z�M�A�B�N�L�M�Z�f�)�,�5�@�;�5�-�)�$�����'�)�)�)�)�)�)�������������$�(�$�������������������������������������ŹŮůŹſ������������������(�*�*�*�"��������������������S�_�l�x�|�x�w�l�g�_�S�F�:�0�1�:�=�F�O�Sù�����!�����ùì×�n�B�1�H�U�nÇàù���������������������������x�s�f�e�u������(�A�N�g�l�j�g�d�^�Z�N�A�5�(�$������"�/�T�a�]�Z�I�;�/�"�	��������������������ûл�����������лûû��������������A�N�Z�g�o�s�z�t�s�g�Z�W�Q�N�A�A�A�A�A�A�Y�e�r�~���������ĺպ��������r�b�Y�I�Q�Y���������������׺���������������� ����������߻ݻ����������������������������������빶�ù̹ܹ��������������ܹϹ��������������'�(�-�'�&���������
�������������������������z�m�h�j�q�}���������.�;�G�T�`�m�y�{�z�v�m�`�T�G�;�5�.�%�%�.�����	����	��������۾������������������������������������ƚƳ������������������ƧƚƁ�x�{ƂƇƒƚ�s�������������������������{�s�p�k�j�s�s����������������������������������T�a�h�m�v�z�������z�m�a�W�T�G�<�;�?�H�T�Z�f�s�s�u�s�f�a�Z�R�M�K�M�W�Z�Z�Z�Z�Z�Z�������
���
����������������������������	��"�-�.�1�.�"��	������������������������������������������������������²®¦¦«²³¹¸²²²²²²��%�(�4�>�<�4�4�(�(�����������g�t¦·��¾²¦�t�i�`�T�O�[�g�����#������� �������������ʾ׾���׾;ʾ�������������������������������������������������������������������������������ŹŭūšŦŭŷŹ�������I�I�F�I�M�U�Z�b�n�{�~ŅŁ�{�z�n�b�U�I�IEuE�E�E�E�E�E�E�E�E�E�E�E�E�E|EuErEkEuEu�����ֽ�:�D�K�I�8�!���ּ������u�o�x���:�F�S�_�l�r�n�l�_�S�H�F�:�6�:�:�:�:�:�:E7ECEPEWEXEPECE7E1E,E7E7E7E7E7E7E7E7E7E7�лܻ�����ܻлû����������ûллл�ǡǣǭǶǷǭǭǡǔǈǂǁǈǋǔǠǡǡǡǡ����*�.�4�*����������������x�r�l�c�l�x��������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD�D�D�D� 2 a � V k < r / u S   | \ E M I P M I < R * I 4 + > ] H ( 8 8 T b > Q W E : = F j B 3 ) V B 9 @ [ \ u 4 W @ � c 3 + = X Y + ! - 3 = C    �  l  (  �  N  a  !  �  !  �  �  w  �  _  @  �    l  �    �  �    �    /  �  q  �    1  �  �  �  -    �  �  s  �  �  �  �     s  �  b  �  �  %  �  M  �  �  �  T  #  0  [  8  �  �  	  �  �  Q  ��C��D���D��;��
;o;�o<o=}�<�1>J=ix�<���>J='�<�t�<���<�j<�j<ě�=u=,1<�h=0 �=t�=+=8Q�>��=�hs=��=���=�7L=t�=]/=,1=@�=H�9=]/=<j=��=�\)=P�`=@�=�-=�%=L��=��T=Y�=}�=q��=]/=�7L=�{=���=���=�+=�t�=�v�=ȴ9=�h>ix�=���=�;d>%>   >
=q>�>S��B#��B	OWB!�.B��BڞB��B'��B��B"�B�B^BbB?�B	[SBz5B/:BȚA�"eB��B{bB|UB�HA�b@B�B��B	*BJXB�B�wB	p�B��B�hB]HB*�B!��BNmB�-B!ݩB��BR5BK�BdB�BS�B$A�tB�LB�B"�B�~B��BƟB]sB��B$ϝB*b9A���B��B��B[�BL�Bg�B��B"�B�B,�BXaB#�NB	0�B"?�B��B�B�B'@�B�QB"@5BHB?|B�jB�B	9sB�$B/?
B��A���B��B�XB��B<6A���BĹB��B�fB>�B�nB��B	@�B�=B��B?eBH�B"?�BA�B��B!��BդB@�B.�B
�B��B@TB$]�A�{0B�JB�,B9�B��B�OB�gB��B�tB$B*4KA�k�B��B�NBB?�B�|B}cB=�B��B,��B�3@��A�D@�<?�F�A��@���@��8AFd�A4߄A�աA���A�HA��A��Ah$Ak��@y �A�$Ac��A��A?�GA��zB��A���A�6�@��$AΈA�pA�eA��,@�`sA�2�@�
@RD�@��!A�H>�V?|�&A�/AgޯAX��A�	�B@A�KA'�A�RdA@D�A��SA[�sAt�eA��"A5]A��A�G�AO��A!�MA�E�A��kC�pAlI@�VYC��w@�~B�VA�$�@��UC��J@���A�l�@4?��BAֈp@�@��AF�TA5 �A�RdAޓ~A�}�A��A�o�AAk l@y�A���Ae	�A²qA? �A��B�A�|�A�~�@�˩AΡ�A�ŰA�{�A�o@��3A��%@�#@P̥@��nA�~>W�?pk�A��SAh�rAX(GA�.B@�A�f.A˼A�|ZA?�A��~A[�Atm&A�z=A4��A�{�A�~�AN�A!�A�Z�A��C��A0E@�.C���@���B��A�2c@�{,C�؎         	      
         ;      v   /      q                     (               
      ~   .   (   3   %                     	   >      
      ,         $         	      	      &         	         %   �                     H                        '      C         E                                          A   +      +   )      !                  '                                                                5                                                   5         #                                                   '         !                  !                                                                /                     NŎ�N!�N>F�NxwN qgNZ�OiqOc�)O%қPW��O��N��}O�e�OxxN&��N���N���NZ1TN�f�N���N��9N�l=O)��N� N�MN�`�O`��O{��O.OP ��O�juN5�O��	N�.�N��NYN/O��N��O��Ov�Nx��N�d�OC�+O �NFg�O5]Ng�N�.GN��NEN&��O�O���N-*N��N��O�O��N�&�P*�N~9kN��@N�VfN���N���NG��O0fu    �  �  �  �  �  <  �  Q  
�  �  G  
L  k  �  z    8  $  %  O  E  Z  �    ]  
�  a  ?  �  �  o  C      �      �  �  �  �    >  �  !  �  y  �  -  �  n  �  ?  �  �  w  �  x     1  !  �  �  �  �  &��1��C��t��ě���`B��`B��o<�<o=8Q�<�h<49X=y�#<�t�<u<u<�o<�C�<�C�=#�
<�/<�j<�`B<���<ě�<���=��=0 �=t�<��=,1=+=o=o=o=��=�w=��=L��=8Q�=,1=0 �=u=8Q�=8Q�=L��=H�9=P�`=P�`=T��=q��=y�#=}�=�+=�%=�o=���=��w=�E�=��=�9X=�j=Ƨ�=���=�F=�>C�������

��������W[gt||tg_[WWWWWWWWWW��������������������qmt{�������tqqqqqqqq����������������������������������������5:BDIbln{����{nbUI<5c^]^clt�����������hc��������������������������*,$�����������������������������
#/<A></+#
��������)-56<?5)�KDFKN[gt�������tlgNK&()6=?6-)&&&&&&&&&&*+2/*$NOQ[_ht���th[ONNNNgfimz{~~zsmgggggggg)257=654)$~~��������������~~~~����

�������������

����������`adimyz|�������zmja`��������������������������������������� �����������������.'$#*/<@HSZ_cc_UH</.��������������������267:B[g������tg[NE;2B=?BO[ht�����th[OICB��������������������
	 )6BJLD>;6.)30.6BMOPOLIB>6333333��������������������;9<AHTU^WUH<;;;;;;;;������������������������������������������������!#�����

#'--*%#

##$$#��������������������)5>BCEEBB5)�����������������������



��������� 	"/;<A?;/+"	����������������������������������������������������������������������UH??<///2/./<HUUUUUU��������������������������(,--$����FGHOU`afa_UHFFFFFFFF	
#$)$#
	��������������������a`amuz������zmga`_aa
��������
��������������������������������������xqnrwz������{zxxxxxx��������������������UWacnrz~��}znaaYUUUU���
"#'(&#
�����,(),/<HHG></,,,,,,,,������������������������������ ������ֻ:�F�S�_�d�e�_�]�S�F�:�3�6�7�:�:�:�:�:�:���������������������������������������������������������������������������������L�Y�a�e�h�e�\�Y�V�L�J�K�L�L�L�L�L�L�L�L�����)�6�:�6�*�)�������������'�!�������������������������������üƼ��������������x�s�s��f�s�����������������������s�i�f�_�]�f�4�A�N�N�H�A�4�%�������������(�4�����	�"�>�I�P�R�L�;�"�	����������������āčĚĦĦİİĪĦĚčā�w�t�p�t�u�āā����&�"�!�!�&���������	���������#�<�N�W�X�Q�I�0�#�����������������O�[�h�t�{Ā�z�q�j�h�\�O�B�;�*�/�6�8�K�O�ݿ�����������ݿܿݿݿݿݿݿݿݿݿݿݿ`�m�y�z���������y�m�f�`�]�]�`�`�`�`�`�`�-�.�:�C�F�O�P�F�C�:�/�-�'�$�&�,�-�-�-�-���������������������������������������˿.�;�G�N�T�[�T�K�G�;�.�.�.�5�.�"�.�.�.�.�/�<�C�H�O�J�H�=�<�/�%�#���#�*�/�/�/�/�Z�f�l�r�n�f�^�Z�M�L�H�M�S�Y�Z�Z�Z�Z�Z�Z�)�,�5�@�;�5�-�)�$�����'�)�)�)�)�)�)����������������������������������������������������ŹůŰŹ��������������������(�*�*�*�"��������������������S�_�e�l�x�{�x�u�l�d�_�Q�F�:�4�:�?�F�Q�Sìù��������������������üùëàÞàèì�������������������������������v�r�y����(�5�A�N�Z�Z�b�_�Z�V�N�A�5�,�(� ��&�(�(��/�H�T�]�X�X�H�;�/�"�	����������������ûлܻ������лû������������������Z�g�k�s�v�s�l�g�a�Z�W�R�Z�Z�Z�Z�Z�Z�Z�Z�Y�e�r�~���������ĺպ��������r�b�Y�I�Q�Y���������������׺���������������� ����������߻ݻ������������������������������������빺�ùҹܹ����������ܹڹϹù��������������'�(�-�'�&���������
�����������������������������y�r�q�v���������T�`�m�y�y�y�y�s�m�`�T�G�;�8�1�;�G�G�T�T�����	����	��������۾������������������������������������ƧƳ����������������ƳƧƚƙƑƑƕƚƤƧ�s�������������������������{�s�p�k�j�s�s����������������������������������H�T�a�e�m�s�z�}���{�z�m�a�Z�J�H�@�>�B�H�Z�f�s�s�u�s�f�a�Z�R�M�K�M�W�Z�Z�Z�Z�Z�Z�������
���
����������������������������	��"�-�.�1�.�"��	������������������������������������������������������²®¦¦«²³¹¸²²²²²²��%�(�4�>�<�4�4�(�(�����������g�t¦·��¾²¦�t�i�`�T�O�[�g����� ������ � �����������ʾ׾���׾;ʾ�������������������������������������������������������������������������������ŹŭūšŦŭŷŹ�������I�I�F�I�M�U�Z�b�n�{�~ŅŁ�{�z�n�b�U�I�IE�E�E�E�E�E�E�E�E�E�E�E�ExE}E�E�E�E�E�E������ּ��.�=�?�.�!���ּ������{�w����:�F�S�_�l�r�n�l�_�S�H�F�:�6�:�:�:�:�:�:E7ECEPEWEXEPECE7E1E,E7E7E7E7E7E7E7E7E7E7�лܻ�����ܻлû����������ûллл�ǡǣǭǶǷǭǭǡǔǈǂǁǈǋǔǠǡǡǡǡ����*�.�4�*����������������x�r�l�c�l�x��������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD�D�D�D� 2 a y C k < r ' r Q  | ? F M I P M I % L * > 0 + 3 . '  6 : F b > Q 6 8 : 5 < j B  ) V C 9 @ [ \ u 4 W @ � c 3 + , V Y + ! - 3 = C    �  l  �  C  N  a  !  �  �  �  N  w  b    @  �    l  �  �  �  �  �  �      �  �  M  �  /  ]  �  �  -  p  [  �  �  '  �  �  �     s  �  b  �  �  %  �  M  �  S  �  T  #  0  �  y  �  �  	  �  �  Q    C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C   C       �  �  �  �  �  �  �  �  �  �  �  w  k  a  X  O  F  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �        %  ,  2  �  �  �  �  �  �      �  �  �  �  �  z  `  K  8    �  �  �  �  �  �  �  �  �  �  �  �    S  #  �  �  �  S    �  �  �  �  �  �         /  c  n  Z  F  0       �  �  �  �  x  �  �  �  �  m  T  9       �  �  �  �  �  z  N     �  �  �  <  +      �  �  �  �  �  �        �  �  �  \  %   �   �  �  �    ;  [  x  �  �  �  �  �  �  l  *  �  �    e  �  �  O  P  I  =  0      �  �  �  �  �  �  d  9  �  �  �  [  A  �  	�  
;  
�  
�  
�  
�  
�  
�  
�  
a  	�  	p  �  �  �  �  �  �  �  �  �    V  �  �  �  �  �  �  �  m  3  �  �  �        �  G  9  )      �  �  �  �  �  �  �  �  �  �  g  C  �  �  �  �  �  	;  	f  	�  	�  
  
D  
K  
9  
  	�  	x  	  �    F    �    [  ^  j  ^  K  5    �  �  �  �  S  %     �  v  '  �  N  �  �  �  �  �  �  �  �  y  l  `  T  G  ;  /  %         �   �  z  x  u  r  o  k  c  \  T  L  B  4  &       �   �   �   |   W      
    �  �  �  �  �  �  �  �  �  �  �  z  n  Y  A  (  8  .  #      �  �  �  �  �  �  x  Z  ;    �  �  �  �  k  $    �  �  �  �  �  �  	  �  �  �  �  �    c  G  )     �  ?  	  o  �  �  �      "  $      �  r    �  2  �  D  �  �  �  �  �    \  �  �  �  ~  x  i  W  >    �  .  �  �  +  E  4  $    �  �  �  �  �  �  e  G  )    �  �  �  �  �  �    B  P  X  Z  N  <  )    �  �  �  �  |  H  �  �  -  �  +  �  �  �  �  �  �  �  �  �  �  �  �  x  `  D  �  �  !  �  %    �  �  �  �  �  �  �  �  �  �  �  �  d  (  �  �  5   �   y  M  \  ]  X  M  >  ,    �  �  �  }  A     �  ^  �  S  �    m  �  	D  	�  	�  
  
Y  
�  
�  
�  
�  
�  
�  
�  
c  	�  	  �  �  �  K  ^  i  �    ;  W  a  U  8    �  �  �  :  �  <  }  �  +  �      .  9  ?  <  4  %    �  �  ?  �  j  �  �  �  S  �  �  �  �  �  �  �  �  r  m  a  L  2    �  �  y  �  r  B  �  s  �  �  �  �  �  �  �  �  m  ,  �  m  �  �    H  E  n  I  j  k  l  m  n  n  g  `  Y  R  G  8  )       �   �   �   �   �  C  %    �  �  �  �  �  �  �  �  �  |  B  �  �  9  �  v   �            �  �  �  �  �  �  �  w  a  L  9      �  �        �  �  �  �  �  �  �  d  P  3  7  �  �  �  �  s  S  D  M  �  �  �  �  �  �  �  �  r  9  �  �  [    �  Y  �  �  �    �  �  �  i  1  �  �  �  ]  <    	  �  �  �  S    �        �  �  �  �  �  �  �  y  W  3  �  �  �  v  T  :  !  �  �  �  �  �  �  �  �  �  ^  &  �  �  -  �    �  �  8  �  P  O  �  v  `  G  2       �  �  �  �  \    �  ,  �  �  X  �  �  �    k  I  #  �  �  �  {  P  %  �  �  �  b  '  �  �  �  �  �  |  p  d  Y  N  B  7  -  "        �  �  �  �  �  x  �  �  �  �            �  �  �  J  �  �  3  �  .  �  >  7  -  $        �  �  �  n  >  	  �    *  �  r    �  �  �  �  �  �  �  �  �  �  p  Z  D  /      �  �  �  }  T      !      
  �  �  �  m  (  �  �  9  �  y  �       !  �  �  �  �  �  �  �  �  �  �  �  �  v  c  P  A  2  $      y  _  D  %    �  �  �  �  h  P  7      �      3  T  v  �  �  �  �  �  �  �  �  u  h  [  O  C  :  0  (  "      �  -  %             �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  0  �  �  �  P    �  �  �  �  �  �  U    �  y  &  �  n  ]  M  <  )      �  �  �  �  I    �  s  !  �  s    �  �  �  �  u  U  -  �  �  m    �    �  �  M    �  �  �  �    "  2  7  ;  >  ?  :  *    �  �  �  @  �  �  >  �  �    �  �  �  �  �  �  �  �  �  �  z  q  g  U  2    �  �  �  �  �  �  �  �  �  |  p  c  N  7       �  �  �  �  �  �  �  �  w  -      �  �  �  �  |  Y  7    �  �  �  �  v  Y  8    �  t  R  .  	  �  �  t  <  �  �  ]  
  �  I  �  {  .  �  k  �    H  d  v  w  t  j  W  >    �  q  �  O  �  �  =  �  ,  �  �      �  |  4  "  �  �  �  3  Z  �  =  
�  	�  (  X  <  1  ,       �  �  �  �  �  h  =    �  �  j     �  �  :   �  !  �  �  �  U  !  �  �  �  N    �  t     �  �  t  M  )     �  �  �  v  [  $  �  �  �  L     �  c  	  �  M  �  �    �  �  �  s  L  ,  �  �  �  m  "  �  �  5  �  ~    �  D  �  �  �  �  �  �  ~  `  >    �  �  �  s  ?    �  �  (  �  S  �  �  �  �  �  m  V  ?  (    �  �  �  �  �  y  ]  B  &      &  �  �  M  �  �  <  �  }  �  $  [  �  �  s  
D  �  �  �  ,