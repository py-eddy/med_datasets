CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�333333      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�^�   max       P�0      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =�v�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E�z�G�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=q    max       @vW�z�H     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @���          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >$�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�^3   max       B,��      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?>Q7   max       C�lo      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?J�   max       C�W�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�^�   max       P@      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�a|�Q   max       ?�y��(      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =��      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�\(�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=q    max       @vUG�z�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�9�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?�X�e+�     �  Pl      	            C               /   	      0   K         0         "      &      %                     %         
         
            2         
                     E         .   'O&�N��N�D}NJ|N��LPPy�OF?N\x�N*xcN}�fP�0N%��O�l�P[��P;=UO�e|N�e�O��N;��N�] O-�N��UOnHN�WO��N�Nכ�O��N��N�gOe�O��N�J�Nj#�N���N�-8N�d�NڰO�(Ot�5M�^�P#WN�84O>sN�wuN�9O�O5�N���Ou�O#O���N�7�N���OzhN�0��h��1��1����ě����
���
��o�D��:�o;�o;��
;�`B;�`B<e`B<u<�o<�C�<�t�<���<��
<�1<�1<�1<�1<�9X<�9X<�j<�j<ě�<ě�<���<���<���<�/<�/<�`B<�h<��<��<��=+=\)=\)=t�=t�=�w=,1=49X=L��=Y�=]/=m�h=}�=��P=�v�d`__agjt|�������tpgdLNPQ[gistvtg[NLLLLLLnjqt�������tnnnnnnnn,).07<CEC?<0,,,,,,,,��������������������UWVH/#
�������
#/NU&)5BEMNE?5/)����������������������������������������)5?BEB5+))����MUn����nU#������������������������������������������������6O[d_OB6����������������������	";HR^]OF;/"	��������������������rt���������������}vr����������������������������������������!#'/<HSU\[US=<7/*%#!��������������������"+/9<EHQTTQMI?/#;:BBOZSOGB;;;;;;;;;;i�����������������ti��������������������+0<=@IUZbgfcbWULI<3+�������
!*0/#
������������������������U[htwxtha[UUUUUUUUUUqtw|�������������wtq��������������������
#*/000/+#
../3<AHJJHA<3/......)-220))#���������������������������	������������������������������854013;?HIRTZ^^VTH;8	
)5;??@?:5)����������������������������(*)%��

"!�~������������������������������������������������������������):EMMJB5)����".;HJQPNHD;/"��������������������)6;BR[^[OE;6)�����������������������������������kheehmz{���zzmkkkkkk{���������������{{{{��������������������������������������n�zÇÓàëìöìêàÓÇ�z�n�d�^�a�i�n�����������������������������������������L�Y�e�l�r�l�e�Y�L�K�B�F�L�L�L�L�L�L�L�L����������������������������������������������������������������m�G�;�,� ��&�'�.�T�`���������������y�m�6�B�O�[�h�m�t�x�|�y�t�h�[�O�B�:�6�2�.�6���������	������������������������|�z�m�m�a�_�`�a�c�m�z���������������Z�c�f�k�p�f�^�Z�P�N�M�M�M�X�Z�Z�Z�Z�Z�Z���A�K�[�s�}�l�Z�N�A�(���	�����������������������������������������������a�z���������������z�a�T�H�0�2�2�;�H�Q�a���+�;�>�>�4�&�	����������������������5�N�g�t�v�t�[�(�����������������)�5�"�/�;�T�a�e�o�y�q�a�H�;�"���������"���*�/�6�:�=�6�*������������B�O�[�`�^�X�O�K�B�<�6�.�����%�)�6�B���ʼּݼڼּʼ�������������������������àìíùþ������úùìàÕÖÖÔàààà��	������������������������������������(�,�/�(��������ݿҿݿ��D�EEEE*E7ECEKEPETECE7E*EED�D�D�D�D��ܻ�����������ܻػܻܻܻܻܻܻܻܻܻܻ����'�4�B�G�@�4�����߻ܻ�ܻѻԻݻ�������!�&�$�!��������������������������������������������������{�x������	�"�-�H�R�V�V�B�/�"��	������������(�4�A�M�P�Z�]�]�Z�M�A�?�4�(�'�'�(�(�(�(�ݿ��߿ݿѿοοѿտݿݿݿݿݿݿݿݿݿݾf�����������������s�f�[�M�H�H�N�Z�^�f���ݽ����#������齷�������z�|��������"�(�4�7�?�A�G�A�4�(�$���������f�s�x�����u�s�f�e�Z�R�Z�`�f�f�f�f�f�f�����	���	�����׾־׾ؾ�������#�/�<�H�S�T�H�C�<�/�#�"��"�#�#�#�#�#�#��#�/�<�>�=�<�/�#������������s���������������������s�n�i�j�s�s�s�s���(�5�A�N�W�N�D�A�5�3�(���������)�5�B�N�Z�g�q�t�n�g�[�N�B�5�)�#��� �)�M�Z�[�Z�W�M�A�4�/�4�A�B�M�M�M�M�M�M�M�M�	�/�H�a�z���������x�m�T�H�"���������	ƧƳƵ����������������������ƳƧƢƤƧƧ�������Ľнݽ�ݽнͽĽ�����������������āčĚĦĳĳĿ��ĿĳĦĚčĄā�~āāāā�y�����������������y�n�l�i�`�l�v�y�y�y�y�#�<�I�a�p�p�b�U�<�0�#����������
�#�
��"�#�0�7�<�C�A�<�0�#���
��	��	�
���
�������
���������������������~�����������ú������������r�g�Y�`�e�m�~ìù��������������óìàÓÇÄÇÐáèì�������ʼ߼����ּʼ�����������������ŠŭŹ������������ŹŭŠŠŝŠŠŠŠŠŠǭǸǵǷǭǦǡǔǈǇǀǈǑǔǡǣǭǭǭǭ�4�M�T�M�G�=�/�'������߻�������4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ? 9 # J M * > B L D l J . E @ , / 1 ) 3 ! w W _ V , h 8 > _ , u I 3 L 2 - 7 \ , a 4 � ` W : = 1 g ` f 3 S M o W  u  �  �  o  �  _  �  �  i  �  >  T    �  m    �    L  �  y  Q    C  @  �    �  �  ^  �  I  �  g  �  �  �  �  ~  �  .  �    p    �  2  �  �  W  �  h  �  �    ��o�D���ě��e`B�o=y�#;��
$�  ;D��;��
=L��<T��<���=]/=�E�=#�
<�9X=�o<���<��=]/<�`B=q��<���=m�h=+<�h='�=+<���=�w=y�#<��<�=�P=0 �=��=��=0 �=D��=t�=��='�=#�
=8Q�=@�=}�=H�9=L��=��w=�O�=��m=�+=���=�F>$�B	ΏB	aB��B&>B�B�!B�<B��B �\B��Bn@B"��B1�B*�B�vA���BNfB3B#�B!��BǺB[3B��Bs#B ��B #�B&�NB{B"%&B�$B<oB �VB�GB�TB�jB��B 2B��A�A�B��B(�B%ZB�mB)��B�B,��B�A��>Bi�B�aB��B�A��SB
�OB� B�B	�LB	0MB��B&4lB?�BP�B>�B��B �4B?B�B"g(B;�BU
B?LA�^3B?�B<�B#6`B!�FB�xB:�B@�BA�B!A�B UXB&��B��B">�B�PB=|B @qB�hB�B=�B�uB�B��A�ğB��B9�BD�B��B)N B1VB,��B�?A�/;B?RB?�B@~B��A��6B?�B��B��AɕUA� (?ן)A/ӟA�F�AjS7A�L?>Q7A���A?��A��6@�A��A�YQA�eA�c�A��8A�ٟ@��A�EA�V�A��\C�lo@��|@���@\� @�r�A�&�A;�A|�aAC�;A(�A7�ABAW�,A�-,A��FAFVeA�lfA�ݶA<�A���B�HA$��A��!AC�A��zA�`�A��z@�gA̓H@�7�A���B��@�*�C���AȔ�A��X?���A/��A��Aj�sAښu?J�A��uA?4A�(�@r�A��xA�pA���A���A�z�A��@���A�k�A�~A���C�W�@�ل@ŝ�@\.�@�P!A��A:��A}�ACW<A(�A7�uAB�AW��A��A��vAE�A��A�'A<�!A���B�@A$\�A��VA �A�q�A�x?A�w�@o�A̰*@�;A���BZ�@��C�ו      
            C               /   	      0   L         0         #      '      &                     %                  
            2         
                     F   	      .   '                  0               A         1   /   #                           '         #            #                              '               %                                                            )               #                                                                                             %                           O�NV��N3NJ|N��LO��OF?N\x�N*xcN}�fP@N%��O��)O���O�� O�N�e�N�51N;��N���N�'�N��UOJ?�N�WO�>)N�Nכ�O��Nf��N�gON�5O5ˇN�J�Nj#�N���N�1�N�d�NڰO�(ON҉M�^�O�P%Nm�BO>sN�wuN�@�O�O5�N���Ou�O#O|2�N�7�N���O4��N��  [  @  f     g  %  C  H  ,  �  %  5  M  �  �  �  K  �  ~  �  a  �  �  >  �  �  �  �  �  �  �  �  T  �  �  �  .  �  
  �  �  �  �  �  �  �    �  �  �    
3  �  �  	�  ��`B���㼃o����ě�<������
��o�D��:�o<ě�;��
<o<���=#�
<�C�<�o=t�<�t�<��
=+<�1<ě�<�1<�<�9X<�9X<�/<���<ě�<���=�P<���<���<�/<�<�`B<�h<��=+<��=,1=t�=\)=t�=�w=�w=,1=49X=L��=Y�=�o=m�h=}�=��T=��ea``cgotw������xtrgePSW[agotttg[PPPPPPPPsntw�������tssssssss,).07<CEC?<0,,,,,,,,�������������������������
/<BGHD</#
�&)5BEMNE?5/)����������������������������������������)5?BEB5+))����/H[gaH?2#�������������������������������������������������)BPUTQJB6)���������������������	
";FO\[UHC;/"
	��������������������������������������������������������������������������������-+./<HHPMH</--------��������������������#,/2;<HOQOKG=</#;:BBOZSOGB;;;;;;;;;;����������������������������������������+0<=@IUZbgfcbWULI<3+������
! 
�������������������������U[htwxtha[UUUUUUUUUUrux}�������������ytr��������������������
#*/000/+#
../3<AHJJHA<3/......)-220))#���������������������������	������������������������������854013;?HIRTZ^^VTH;8
)59<=>=85)
��������������������������"%$ ����! �~������������������������������������������������������������):EMMJB5)����".;HJQPNHD;/"��������������������)6;BR[^[OE;6)����������������������������	�������kheehmz{���zzmkkkkkk{���������������{{{{��������������������������������������n�zÇÓàéìðìèàÇ�z�n�e�a�`�a�k�n�����������������������������������������L�Y�c�e�k�e�a�Y�T�L�G�K�L�L�L�L�L�L�L�L����������������������������������������������������������������`�m�y�����������y�m�`�T�G�?�;�<�B�K�T�`�6�B�O�[�h�m�t�x�|�y�t�h�[�O�B�:�6�2�.�6���������	������������������������|�z�m�m�a�_�`�a�c�m�z���������������Z�c�f�k�p�f�^�Z�P�N�M�M�M�X�Z�Z�Z�Z�Z�Z���5�B�D�O�S�N�A�5�������������������������������������������������������a�m�z�����������z�m�a�T�H�6�5�5�;�H�S�a�����	�"�/�1�/�%��	���������������������5�B�N�Z�b�g�c�U�B�5�)����������)�5�/�;�T�a�m�v�m�a�H�;�/����������"�/���*�/�6�:�=�6�*������������)�6�B�O�R�R�O�L�B�6�)�$� �%�)�)�)�)�)�)���ʼּݼڼּʼ�������������������������àèìùý������úùìàÖ×Ø×àààà�����������������������������������������(�,�/�(��������ݿҿݿ��D�EEEE*E3E7ECEEECE7E*EED�D�D�D�D�D��ܻ�����������ܻػܻܻܻܻܻܻܻܻܻܼ�
��'�4�;�>�>�4�'������������������!�&�$�!��������������������������������������������������{�x����	��#�7�H�K�F�;�/�"���	� �����������	�4�A�K�M�Y�W�M�D�A�4�*�.�4�4�4�4�4�4�4�4�ݿ��߿ݿѿοοѿտݿݿݿݿݿݿݿݿݿݾf�����������������s�f�_�Z�O�J�P�Z�`�f�нݽ��������ݽĽ������������������о�"�(�4�7�?�A�G�A�4�(�$���������f�s�x�����u�s�f�e�Z�R�Z�`�f�f�f�f�f�f�����	���	�����׾־׾ؾ�������/�<�H�O�O�H�@�<�/�%�#�"�#�&�/�/�/�/�/�/��#�/�<�>�=�<�/�#������������s���������������������s�n�i�j�s�s�s�s���(�5�A�N�W�N�D�A�5�3�(���������)�5�B�N�S�g�m�q�k�g�[�N�B�5�)�%�!� �%�)�M�Z�[�Z�W�M�A�4�/�4�A�B�M�M�M�M�M�M�M�M�/�;�H�a�q�x�v�o�b�T�H�"������
��/ƳƳ������������������ƳƧƤƧƱƳƳƳƳ�������Ľнݽ�ݽнͽĽ�����������������āčĚĦĳĳĿ��ĿĳĦĚčĄā�~āāāā�����������������y�t�o�l�i�l�y�}���������#�<�I�a�p�p�b�U�<�0�#����������
�#�
��"�#�0�7�<�C�A�<�0�#���
��	��	�
���
�������
���������������������~�����������ú������������r�g�Y�`�e�m�~ìù��������������óìàÓÇÄÇÐáèì�����ʼּڼ���ּʼ�������������������ŠŭŹ������������ŹŭŠŠŝŠŠŠŠŠŠǭǸǵǷǭǦǡǔǈǇǀǈǑǔǡǣǭǭǭǭ��'�4�@�G�C�9�,�'���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� = A * J M : > B L D g J / 1 - * / $ ) 6  w N _ / , h * A _ " f I 3 L 2 - 7 \ 0 a ' � ` W C = 1 g ` f ( S M ^ W  i  �  G  o  �  `  �  �  i  �  �  T  N  �  �  �  �  �  L  �  �  Q  �  C  6  �      }  ^  �  �  �  g  �  �  �  �  ~  �  .  �  �  p    �  2  �  �  W  �  �  �  �  �    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  P  X  Y  T  H  ;  .      �  �  �  �  }  m  _  O  %  �  �  $  .  7  ;  ?  ?  =  7  /  &        �  �  �  �  c  ,  �    $  5  D  S  _  d  c  Z  J  0  
  �  �  k  .  �  �  �  e           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  ^  T  K  A  8  -  #       �   �   �   �   �   �   �   q   ]   I  �  G  ~  �  �  )  �  �    $    �  �  g  �  [  �  �  �  �  C  9  /  '                    �  �  �  �  �  �  �  H  B  <  5  /  (      	  �  �  �  �  �  �  �  �  {  d  M  ,  "      �  �  �  �  \  6    �  �  �  r  N  *     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    q  �  �         
      #  !      �  �  �  p  :  �  o  �  5  5  4  /  '          �  �  �  �  �  �  �  �  �  }  f  K  M  K  G  ;  3  ;  >  >  )    �  �  �  �  t  O  0   �   �  �    5  ]  x  �  �  �  �  �  �  �  R    �  �  /  �  �   �  y  �  (  S  q  �  �  �  �  g  :    �  �  }  !  �  �    Q  �  �  �  �  �  �  �  w  E    �  �  =  �  �  �  _    �  :  K  I  H  F  G  H  J  B  7  +      �  �  �  �  �  �  l  S  K  �  �    2  ^  �  �  �  �  �  h  I    �  q  �  �    �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  �  �  �  {  o  a  R  @  ,    �  �  �  �  �  -  h  )  �  �  o  �  �  �  !  ?  T  _  `  Z  K  (  �  �  U  �  �    �  2  �  �  �  �  p  [  F  0    %  5  A  4  '    �  �  �  �  f  �  �  �  �  �  �  e  6    �  �  Q  �  �  F  �    -  7  3  >  7  1  *  #        �  �  �  �  �  x  \  ?  "    �  �  �  �  �  �  �  �  �  �  �  u  U  4      �  �  �  ,  S  ^  �  �  �  �  �  r  ^  J  1    �  �  �  �  j  3  �  �  !  �  �  �  �  �  }  t  l  e  ^  R  E  7      �  �  }  X  :    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  \  5    �  Q  �  �  �  �  �  �  �  �  �  �  �  p  C    �  �  n  7   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  n  e  \  R  I  �  �  �  �  �  �  �  �  e  I  1  !      �  �  �  �  �  Z  X  �  �  �  �  �  �  �  �  �  �  n  @    �  �  Q  �  ?  �  T  M  F  ?  8  -  "      �  �  �  �  �  �  �  �  �  �  z  �  �  �  �  �  �  �  �  �  }  l  X  D  0      �  �  �  �  �  �  �  �  �  �  ~  m  [  H  3      �  �  �  �  X  *  �  �  �  �  �  �  �  �  �  �  �  y  \  <    �  �  �  l  ;    .  )  #          �  �  �  �  �  �  �  �  �  d  C  V  n  �  �  �  �    k  V  K  D  D  H  J  H  @  +    �  �  �  �  
    �  �  �  �  �  �  �  �  �  t  [  C  .    �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  d  8  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  f  Y  K  =  .      �  z  �  �  �  �  �  �  �  e  C    �  �  d    �  U  �       �  �  �  �  �  �  �  �  �  o  Z  C  ,    �  �  p  7  �  �  �  �  �  �  �  �  �  �  �  o  \  J  :  -          �   �   �  �  �  �  �  �  ~  |  x  s  i  Z  J  9  '          I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  M  ,    �        �  �  �  �  �  �  �  p  ^  Q  G  ;  %  �  �  m  4    �  �  �  �  }  l  \  M  ?  0  !      �  �  �  �  �  i  <  �  �  �  �  �  �  �  �  �  �  �  �  �  v  [  @  %    �  �  �  �  �  �  t  L  +  *  x  `  *  �  �  H  �  Q  �  0  �             �  �  �  �  �  �  y  I    �  �  r  5  !  &  w  	�  
  
*  
3  
,  
  	�  	�  	�  	V  	  �  w    �    W  X    *  �  �  �  m  [  I  ;  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  v  \  +    �  �  |  J    �  �  /  �  �  *  �  �  	  	9  	S  	�  	�  	{  	Q  	  �  �  @  	  �  c  �  <    �  �  q  �  �  �  �  d  O  :    
�  
�  
q  
  	�  	  j  �  �  �  �  �