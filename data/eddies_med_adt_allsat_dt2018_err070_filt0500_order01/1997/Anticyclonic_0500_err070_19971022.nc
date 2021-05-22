CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��"��`B      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�:2   max       P�{      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =��#      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Ǯz�H   max       @E������     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vs��Q�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P`           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�Ӏ          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��9X   max       >hr�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�gV   max       B,�^      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�xQ   max       B,�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?)̻   max       C�t8      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?06W   max       C�pT      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ;      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          /      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�:2   max       P:��      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���+j��   max       ?�>BZ�c       �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >O�      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Ǯz�H   max       @E������     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(    max       @vqG�z�     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�`          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D5   max         D5      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?�<�쿱\        R<                  
      	         U   	   1         A      ,      	         �      P            3         
      e      F                     #                  5      #         	      (   !   	N�l�N[�XN���ORzM�:2Nw�N��N�-O3 XN�aCPB�ON�V:O�&�N���OyWPP:mrO P"�2O��UN�d@NwOE� P�{N�[O�n�NP�nN�3�N?l�P[ҾN.��O
7N���P�O��OB�kO� }N�O&O�rN	��N�3N3�%P	*�N�ȦNINۖN��Nm+O��tOyBOd�9OQ�NV��N��N���O���OH�jM���h��`B�u��`B��o%   ;D��;��
;��
<#�
<e`B<�t�<��
<��
<��
<�1<�9X<ě�<ě�<ě�<ě�<���<���<���<�h<�<�<��=+=+=+=\)=�w=�w=�w=0 �=0 �=8Q�=@�=D��=P�`=]/=ix�=ix�=ix�=m�h=q��=u=}�=�hs=��
=��
=���=�1=���=��`=��`=��#c_[gltv����~tgcccccc|v}�������||||||||||��������������������#$/<HUahnxsnaU</$$��������������������MMNOY[^gkkgb[NMMMMMM��������������������Z[[ht{}vth_[ZZZZZZZZru~��������������{trttz~�����������ztttt������8AIHD5)���NNO[gtyytg[NNNNNNNNN %,21<HanvwvqkaUH/( �������������������������������������
#-23/#
�����������������������������,34-��������rs���������������yvr),6?:6)#0<=@<90*#������������������������)LZZS``YB����������������������������������������������  
��������
).45::5)�
!#,#
�������������
5NRNJB)	�������������/.67BO[hhjhb[UOB66//Z[[chmt~|tth[ZZZZZZ��������������������������

���������������������������������

������������������������������������������������	
#$'(('#
����������������������������������������jmoyz{����{zsmjjjjjjbft���������������ib����������������������

�����������������������������|{��������||||||||||igmnoz|}zniiiiiiiiii������

������������

�������|z{|���������������������#),)����$)+,)(rz{������{rrrrrrrrrr������������������������������������������� )06<>62)���������������������������'�)�*�)����������������n�zÇËÌÇ�z�n�k�j�n�n�n�n�n�n�n�n�n�nìù����������������ùìåçìììììì�������������������������������)�6�B�C�B�=�7�6�5�.�)�)�)�)�)�)�)�)�)�)�b�n�t�{ŇŊŇł�{�n�h�b�\�[�b�b�b�b�b�b�����������������������r���������r�f�d�\�f�n�r�r�r�r�r�r�r�r�ܹ�����������ܹٹӹԹϹùϹչ��a�m�t�z�������z�m�a�a�V�T�T�T�]�a�a�a�a���
�#�<�b�p�p�`�I�<�#�
�����������������y�������������y�r�o�o�w�y�y�y�y�y�y�y�y�a�m���������������������m�a�X�Q�P�Q�T�aE�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�Eٿ;�G�T�`�m�y�����{�m�`�T�G�;�"����.�;�)�B�[�q�t�D�7�2�)���������������)�(�4�M�Z�f�s�t�t�k�k�u�s�f�Z�M�A�-��!�(�����������������������z�T�K�M�m�������ž�������������v�f�Z�A�7�7�D�@�E�M�Z�s��'�)�3�6�?�3�,�'�������'�'�'�'�'�'�������������t�r�j�r�v��������ĚĦĳĸĿ��������ĿĦčā�t�r�y�~āčĚ�h�s�j�uċĐā�h�O�)��������������)�B�h���ûлٻۻӻлû�����������������������������)�6�B�O�U�V�V�R�O�B�6�)������������������ݿֿֿݿ��������꿫���ĿͿϿѿֿѿĿĿ�����������������������������������������������������������ƁƚƧ���������������Ơ�b�O�7�C�O�qƁ����*�,�2�*�'�������������ѿݿ��������ݿۿѿ˿Ŀ����ĿſѿѾM�Z�[�f�n�s�w�s�f�Z�M�M�G�K�M�M�M�M�M�M����������������������������������}��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D�D�D�D��������ʾ׾����������׾ʾ��������������"�/�<�A�D�H�G�=�/�"��	�������������n�p�v�t�zÇËÇ�}�z�n�f�a�W�U�U�Y�a�c�n���!�-�4�:�F�Q�S�_�b�Y�N�F�:�!������(�4�A�M�Z�]�f�n�m�f�Z�M�A�4�'�������ĽĽĽ��������������������������������y�����������~�y�n�v�y�y�y�y�y�y�y�y�y�yŠũŭŹźŹűŭũŠŔŒŔŘŠŠŠŠŠŠ�����������������s�Z�N�C�@�?�L�g������������&�'�-�(�����������������m�r�u�m�j�`�T�I�J�T�`�g�m�m�m�m�m�m�m�m�/�/�#��#�/�/�/�<�@�<�/�/�/�/�/�/�/�/�/�(�5�A�C�A�9�5�(�"�#�(�(�(�(�(�(�(�(�(�(�Z�g�s�t�s�k�g�Z�T�O�Z�Z�Z�Z�Z�Z�Z�Z�Z�ZE�E�E�E�E�E�E�E�E�E�E�EtElEmEkEkEuE�E�E����
�������
�����������������������������ɺֺ����ۺֺɺ��������������������������������������������~�|���������g�h�t�u�t�t�g�f�d�f�g�g�g�g��������������������������������ǔǡǭǷǶǭǩǡǔǎǈǆǈǐǔǔǔǔǔǔ�'�4�@�E�G�5������������������'��������	��������ܻԻллֻܻ��E*E7ECEFEDECE7E*E*E*E*E*E*E*E*E*E*E*E*E* % , 4 \ o : 9   8 3 % A *   P ` 6 8 ; X : a . . C J < v _ Z 9 J T Q P 5 X s L � W > X @ R . 0 G ' % L ^ z Y 6 s < 3  �  f  �    <  �  :  �  �    4  �  .  �  �  �  
  �  "  �  �  �  �  �  �  q     �  L  i  E  �  �  \  �      �  f  �  F  T  �    x    4      :  �    �  M  �  �  �  ��9X�D���49X<�o<#�
<#�
;��
<e`B<�<���=ȴ9<���=�C�=#�
='�=�{=P�`=�7L=H�9=o<��=@�>hr�='�=�/=+=t�=��=���=�P=49X=49X=�\)>V=L��=�l�=]/=�%=�+=L��=e`B=m�h=�^5=��=}�=�\)=y�#=�%=�x�=�E�=�x�=��=�E�=�j=�`B>hs>	7L>$�B	�	B
��B��B�kB�B�+B i�B�WBk�B �B�B		Br�BpB�xB	�BB��B�^BSB%��B/BH�B"��BW�B�tBbBqB��BV{B��B�ZB+�B<zB 0�B�<BZiB t�B$��B,�^B,/�A�gVBScB"�B��B<�B��B�BR�B_B��B�B�B) �B#wB��Bz Bx�B	��B
��B��B�B?�B��B dB��Bb�B �QB�dB	C B�+BNBB�BAsB?fB�B�(B=:B%�'B��B?�B"��BѦB��B��B>�B�%B>BI�BԕBDcBA B D�B��B�pB �4B$�3B,�B,9'A�xQB�[B?�B�MB@B�UB�BD�B�BgfBA�B?�B(ŀB'B�yBC�B@A��3AȈ�A��RA��0Aא�A��A.�L@�`?)̻A�d�A�}dAm�iA��C�t8Ae��A�?�A<UDA�ۭAC>$?���@�2�A�>�A�@�I�A�<rA~��Av��AIŻBJA�4A|�PA?uYA�K�C��$AR�A���A�w�@y�oA:�A%UA�A���A�]HA�<�Ah��A�aA�%�A�0C��A���@.�w@1HA�`@X��B�p@ī�@�ªC��A�=1AȁcA��`AҊ�A׀A���A.��@� �?06WA�v�A��Am�DA���C�pTAd�jA��A<��A�֠AC�?��o@�,�A߄kA�~�@��_A׀�A?Aw�AJ��B=�A���A{�A>�A�f�C��7ARJ�A���A�tV@|u�A9AHA% �A�A��GA�o:A��Aj�IA4A�JA�}xC�A���@1�L@��A���@TB��@š @�'�C��                         
         U   	   1         A      ,      	         �      P            3         
       e      G                     #   	               5      #         	      )   !   
                                 +      #         1      *   %            ;                  5            +                              +                                       !      
                                 !               +      !                                 /            )                              %                                       !      
N�l�NB�@N���Nr(�M�:2Nw�N��NT(�N���N�>�O���N�V:O�f�N%��OyWPP/oO��O�B�Oq�N�d@NwOE� O���N}OP�NP�nN�3�N?l�P:��N.��O
7N���Pp0O)�OB�kO���N���N�hOO>9N	��N�3N3�%O�_�N�ȦNINۖN��Nm+O+`OyBO=��OQ�NV��N��N���O���OH�jM�  �    e  �  A    !  �  {  }  	v  A  �    �  �  �  �  �    �    �      �  C  �  �  &  �  [  '  =  M  
4  A  �  0  �    �  $  �  ?  �      
�  �  K    m  W  �  5  �  D��h��/�u;�o��o%   ;D��;ě�<D��<49X=<j<�t�<�h<�`B<��
=t�<�h=�P=o<ě�<ě�<���>O�=o=]/<�<�<��=�P=+=+=\)=#�
=�O�=�w=u=8Q�=T��=D��=D��=P�`=]/=�%=ix�=ix�=m�h=q��=u=��-=�hs=�1=��
=���=�1=���=��`=��`=��#c_[gltv����~tgcccccc}w~�������}}}}}}}}}}��������������������-//<HNTHD<2/--------��������������������MMNOY[^gkkgb[NMMMMMM��������������������[[]htx{ttha[[[[[[[[[{{~���������������{{vwz����������zvvvvvv����)8<>;5)����NNO[gtyytg[NNNNNNNNN--17<HUakprrnkaUH<0-�������������������������������������
#'(#
��������������������������������$$������~������������������~),6?:6)#0<=@<90*#��������������������������)0>A@8)������������������������������������������  
��������
).45::5)�
!#,#
������������)5BMLHB-)������������/.67BO[hhjhb[UOB66//Z[[chmt~|tth[ZZZZZZ�����������
�������������

���������������������������������

������������������������������������������������  
##&((&##
����������������������������������������jmoyz{����{zsmjjjjjjqkt����������������q����������������������

�����������������������������|{��������||||||||||igmnoz|}zniiiiiiiiii������

����������

�������~|}~���������������������#),)����$)+,)(rz{������{rrrrrrrrrr������������������������������������������� )06<>62)���������������������������'�)�*�)����������������n�zÇÊÌÇ�z�n�l�k�n�n�n�n�n�n�n�n�n�nìù����������������ùìåçìììììì������
������������������������������)�6�B�C�B�=�7�6�5�.�)�)�)�)�)�)�)�)�)�)�b�n�t�{ŇŊŇł�{�n�h�b�\�[�b�b�b�b�b�b�����������������������r�r�������r�f�f�^�f�r�r�r�r�r�r�r�r�r�ܹ����������������ݹݹܹ׹ܹ��a�m�q�z�������z�m�b�a�W�U�^�a�a�a�a�a�a�#�<�I�P�V�R�A�0�#��
����������� �
��#�y�������������y�r�o�o�w�y�y�y�y�y�y�y�y�m�����������������������t�m�_�Y�W�Y�a�mE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ;�G�T�`�m�y�����{�m�`�T�G�;�"����.�;�B�[�g�t�u�g�[�7�)����������������)�B�4�A�M�Z�]�f�m�f�d�d�Z�M�A�4�(�&�(�*�2�4���������������������������h�^�d�m�z���������������������s�f�Z�O�T�O�S�Z�f�x��'�)�3�6�?�3�,�'�������'�'�'�'�'�'�������������t�r�j�r�v��������ĚĦĳĸĿ��������ĿĦčā�t�r�y�~āčĚ������)�6�D�M�O�J�6�����������������û̻лһлȻû����������ûûûûûûû��)�6�B�M�N�I�B�8�6�)�&��������"�)������������ݿֿֿݿ��������꿫���ĿͿϿѿֿѿĿĿ�����������������������������������������������������������ƚƧƹ������	��
������Ʀ�u�g�W�O�RƁƚ����*�,�2�*�'�������������ѿݿ��������ݿۿѿ˿Ŀ����ĿſѿѾM�Z�[�f�n�s�w�s�f�Z�M�M�G�K�M�M�M�M�M�M�������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������ʾ׾����������׾ʾ�������������"�/�7�<�?�@�<�/�"�����������������n�s�s�zÇÉÇ�z�z�z�n�c�a�^�W�X�[�a�m�n�-�.�:�F�J�R�F�=�:�9�-�)�!��!�(�-�-�-�-��(�4�A�M�Z�f�l�g�f�Z�M�A�4�(�'�������ĽĽĽ��������������������������������y�����������~�y�n�v�y�y�y�y�y�y�y�y�y�yŠũŭŹźŹűŭũŠŔŒŔŘŠŠŠŠŠŠ�����������������������s�e�Z�V�j������������&�'�-�(�����������������m�r�u�m�j�`�T�I�J�T�`�g�m�m�m�m�m�m�m�m�/�/�#��#�/�/�/�<�@�<�/�/�/�/�/�/�/�/�/�(�5�A�C�A�9�5�(�"�#�(�(�(�(�(�(�(�(�(�(�Z�g�s�t�s�k�g�Z�T�O�Z�Z�Z�Z�Z�Z�Z�Z�Z�ZE�E�E�E�E�E�E�E�E�E�E�E�E|EuEsEvEzE�E�E����
�������
�����������������������������ɺֺ��޺غֺҺɺ��������������������������������������������~�|���������g�h�t�u�t�t�g�f�d�f�g�g�g�g��������������������������������ǔǡǭǷǶǭǩǡǔǎǈǆǈǐǔǔǔǔǔǔ�'�4�@�E�G�5������������������'��������	��������ܻԻллֻܻ��E*E7ECEFEDECE7E*E*E*E*E*E*E*E*E*E*E*E*E* % & 4 / o : 9  ; 6 " A  $ P b  +  X : a - 6 C J < v ^ Z 9 J P ( P 0 \ Q L � W > Q @ R . 0 G $ % G ^ z Y 6 s < 3  �  Q  �  }  <  �  :  _  �  �  �  �  g  ?  �  �  G  �  �  �  �  �  �  :  �  q     �  �  i  E  �  �  8  �  ;  �  �  T  �  F  T  �    x    4    l  :  �    �  M  �  �  �    D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  D5  �  �  �  �  }  u  m  f  _  X  P  I  G  F  E  G  I  P  X  `  �  	       �  �  �  �  �  ^  =  "      �  �  �  �  �  �  e  a  ]  Y  U  Q  J  C  <  5  -  "        �  �  �  �  �  7  N  `  h  f  d  m  x  �  �  �  x  _  C    �  �  |    �  A  K  7  �  �  g  =    �  �  �  b  4    �  �  r  @     �      �  �  �  �  �  �  �  �  b  ;    �  �  �  o  C  &    !                        
         �   �   �   �  �  �  �  �  �  �         �  �  �  �  |  S  *  �  �  �  u  �    <  U  m  y  z  r  c  J  &    �  �  �  D  �  �  4  0  {  |  z  u  o  k  e  \  M  5    �  �  �  x  G    �  f    �  M  �  �  	  	E  	b  	t  	p  	_  	=  	  �  G  �    D  G  4  �  A  ;  5  0  +  '  #                	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  >    �  �  ]  �  p  �  �  �                    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  G  %  �  �  �  s  A     �  �  >  r  �  �  �  �  �  l  *  �  ~  "  �  $  y    8  n  >   �  �  �  �  �  �  �  �  �  �  �  �  �  �  V    �  �  E    �  t  �  �  �  �  �  �  �  �  �  t  E    �  �  7  �  Y  �  ?  8  P  a  l  �  �  �  �  �  �  �  ^  9    �  �  +  �  @   �    �  �  �  �  �  �  �  �  j  B    �  �  �  b  2  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  s  k  f  b  ^  S  E  6    �  �  �  [  &  �  �  �  �  �  x  I    �  �  L  �  �  W  �  �  �    G  ]  ^  B    (  }  �  =  �      �  
�  �  w  �  �  �  �  �            �  �  �  g  D    �  �  �  �  �    u  �  �      �  �  w    �    
�  	�  �    �  �  �  �  �  �  �  �  �  �  �  q  b  S  D  4  %       �   �   �   �  C  >  9  4  6  ;  ?  ;  3  *    
  �  �  �  �  l  H  %    �  �  �  �  �  �  �  �  �  �  �  ~  q  d  ]  a  d  j  s  {  �  �  �  �  �  v  q  �  �  �  �  i  4  �  �  K  �  [  �  1  &  (  *  ,  -  0  2  4  7  9  <  ?  A  D  F  K  Q  W  ]  c  �  �  �  �  �  �  �  �  �  �  �  �  g  D     �  �  �  #   |  [  N  @  3  &              �  �  �  �  �  [  .  �  �    %  !          �  �  �  �  �  [  (  �  �  "  �  Z  �  Y  �  r  �    :  <  -    �  �  W  �  g  �  a  �  
b  �  �  M  B  :  7  7  8  9  :  ;  :  8  2  +      �  �  �  �  O  	�  	�  	�  
  
.  
4  
+  
  	�  	�  	R  	  �  %  �  �  #  7    \  8  <  ?  @  <  5  *                      �  �  �  �  v  n  _  R  M  H  {  �  �  �  �  �  �  �  �  �  i  a  a  ,  0  &      �  �  �  �  t  I    �  �  ?  �  n  �  �  ;  �  �  �  �  �  {  q  h  _  V  M  D  ;  2  *  !         �             0  A  Q  X  P  H  ?  7  -  $        �  �  �  �  w  h  Y  J  8  '      �  �  �  �  �  �  �  �  �  �  "      #  !    
  �  �  �  �  h  +  �  �  �  g  /  �  �  �  �  �  �  �  �  �  �  r  Y  H  >  3       �  �  �  �  �  ?  ;  7  2  ,        �  �  �  �  �  �  ]  ;  ;  D  M  U  �  �  �  �  �  r  ]  B  $  �  �  �  r  A    �  �  k  2   �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  b  T        �  �  �  �  �  �  �  �  �  �  {  l  ]  M  >  /     	�  	�  
/  
[  
y  
�  
~  
j  
F  
  	�  	Y  �  V  �    F  U  *    �  �  y  ^  B  %    �  �  �  |  S  '  �  �  q    �  -   �  4  B  I  E  ;  -      �  �  �  �  T    �  #  �    �      �  �  �  �  k  9    �  �  �  �  �  n    �  D  �  {  &  m  \  J  9  %    �  �  �  �  �  �  �  L  �  �  _    �  T  W  9    �  �  �  Z  #  �  �  �  k  B    �  �  �  �  c  ;  �  �  �  r  I  '    �  �  z  J    �  �  �  Q    �  �  4  5  .  '  �  �      �  �  t  9  �  �  �  4  �  C  �  �  �  �  �  x  [  E  3  %      �  �  �  Y    �  `  �  �  �   �  D    �  �  �  \     �  �  U  	  �  Y  �  �  I  �  �  *  �