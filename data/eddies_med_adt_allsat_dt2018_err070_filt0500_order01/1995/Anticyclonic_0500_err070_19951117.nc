CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N	��   max       P��      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �@�   max       =\      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F5\(�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\)    max       @vf�Q�     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >��+      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��R   max       B0N�      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�9�   max       B0N:      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =J	�   max       C���      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ><�   max       C��(      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N	��   max       P5�      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Fs���   max       ?�@N���      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >@�      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F5\(�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\*    max       @vffffff     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q            d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�<@          �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         At   max         At      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�@N���     P  J            '   [   p      -   "                 
      c   6                M   E   >   !   3         	                  $   +         	            E   x            
O��N�i�O2WP*�EP�<�Pa�%N���PtJ^O�lP��O� �ON��Or�HN��dO
z�O6�LPL�
P� "N��5O�#sN�R�O�O��PS	�PI��Nǽ?O���Oi�O��gO��O��O~TN���N�N!'�O�HCO1�N��N\wTN���NS>N�0Nr[�O��P	��N���N�O�N�-N	��@���o%   ;�`B<D��<T��<T��<e`B<e`B<u<�C�<�C�<�t�<���<�j<ě�<�h<��=o=+=\)=t�=�P=��=��=��=��=#�
='�=<j=<j=@�=@�=@�=@�=L��=L��=aG�=m�h=q��=q��=u=y�#=y�#=y�#=�\)=��P=� �=\KJNPS[gt�������tg[OK���)02)�������205?BN[^bed^[ONCBA52���������
�����3EB7ARht��������[B63����5BHJC5)�����NNR[gty����{tg[YONNN������$)8. ����������):DFB7)���
 )5Ng��������gN
�����������������������������������������{|�������������������������

����������
!)*+)��!)+46BCORUXYZYOB/)!�����#/<?09:5-��������!(BNPV[][TH)�-./4<HJUWVUOIHH<1/-- �� )POTUQCB6)hipt}����������thhhh������������������������
#/9?BB=</#
��)>Ng�������}gB)!����� 
#/5>PXHE/��������� �����������������������������������������������������������������7COTZXOC6*(*777������"*&$�������� 
 ## 
������
"#%&$#
�����

���������

�������������������������
!#%#!
�����������������������/26BEO[d[[OB:6//////����	������������������������55<@HJUVYUTHD<555555������������������������������������������0854)������� 	������"&/;>B@;/"

z{��������zzzzzzzzzz�U�a�q�zÇÍÖÒÊ�z�n�a�\�U�D�=�;�>�H�U�G�T�V�V�W�Y�X�T�G�@�<�;�?�B�G�G�G�G�G�G�{ŇœŠŢŠŗŔŇ�{�n�b�]�U�T�U�V�b�r�{�ûܻ��'�<�L�4��黷�����x�c�h�x�����û���M�f�����Ǽȼ��������Y�4����ջۻ��
�<�U�Z�W�P�E�8�#�
�������������������
������������������������������������������5�>�Q�O�L�A�(�����ݿĿ��������Ŀ���.�;�G�_�h�i�h�^�G�.�"�	����� ���������.��6�[āĐĘĜĖā�t�[�B�6����������������������������������m�f�[�`�d�i�s����(�5�:�<�5�/�(���������������4�A�M�N�N�M�A�4�(������������(�4�U�a�n�z�|�z�v�t�y�n�a�U�L�H�B�H�L�T�U�U������������������������������������Ѿ��������ʾ;׾׾Ҿʾ��������������������N�g§¨¬¢¤�g�N�5�)����5�NƧ�������$�=�I�Q�=�$�������ƚƁ�j�l�tƧ������������������������������������:�F�_�x���w�_�W�-�!������޺����:�-�:�F�O�S�T�S�R�F�>�:�-�)�'�%�%�-�-�-�-�5�A�E�N�R�X�Z�\�^�Z�X�N�F�A�4�(�&�(�/�5D�EEEE"E&E%EEED�D�D�D�D�D�D�D�D�D�����"�6�H�P�L�4�������������������������������/�a�t�z�y�m�a�;���������������D{D}D�D�D�D�D�D�D�D�D{DoDkDbD^DbDoDyD{D{�k���Ϲܹ���ܹϹù������z�w�_�R�F�b�k������/�4�A�L�A�4������ݽڽսݽ��������������������ŹŭŢŋŉŔŠų�����ҿ��������y�m�f�d�e�f�k�m�y���������������y�����������������������y�l�j�f�`�`�l�y�׾����"�-�.�"���	������׾Ӿ;׿`�m�y�~���z�y�m�i�`�T�G�B�<�G�N�T�^�`�`�������ʾ׾ھ׾ʾ������������������������Z�f�s�������s�g�f�Z�W�Z�Z�Z�Z�Z�Z�Z�Zù����������������àÓ�z�v�u�z�}ÇÓàùEuE�E�E�E�E�E�E�E�E�E�E�EzEuEiEhEdEeEiEu�������������������������������������������
�������
�
�����������������������
�
��
����������������������������4�A�M�O�N�M�A�6�4�-�4�4�4�4�4�4�4�4�4�4��(�,�(��������
���������@�M�M�Y�f�f�r�~�r�f�Y�M�L�@�?�;�@�@�@�@���������ʼּ����ּ�����������������r���������ƺǺ��������~�e�L�C�:�B�R�e�r�	���"�&�'�"��	���������������	�	�	�	������
���
���������������������������\�O�C�6�*�&�*�0�6�C�O�\�c�\�\�\�\�\�\�\�Y�e�q�r�z�r�e�Y�X�U�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y   ) f c  7 ' ( # 9 % @ = 6 7  S   i C h , O B Q n j Q [ ! j N a ] 2 # c � p = � t T $ 6 % N F    )  [  �  �  �  �  -  �    �  ]  �    )  ;  �  N  �  �  �  !  ~  $  �  �  �  W  �  �  D    �    W  \  �  v  7  �  �  :  v  �  �  a  �  �  �  4��j:�o<u=<j=��>   =\)=m�h=@�>��+=49X=o=�P=t�=o=�w>o=�1=Y�=�o=0 �=D��=�l�=��=���=�\)=�E�=�+=�\)=aG�=�o=ix�=}�=L��=T��=� �=�v�=u=�%=�C�=�+=�%=�O�>�>8Q�=��
=��T=Ƨ�=���B	��B�xB&�B#ZBvB��B	Q�BF�Bp�B��BF�B�4B��B)\B��B3�BA�B�)B�$B-BզB?�B�B�B8�B�kB�	B#c�B�wB0N�B."B�sB�kB$(�B$IgBDB��B��Br,B_BӞB'�B�~B>7B��B�A��RB�B�7B	�WBBMBP�B#A�B@B>B	?�B=CBAB��BC&B�QB��B>#B_�B?�B=�BN�BȟBu�B-�B@{B��B	;�B�B�#B�B#@�B?�B0N:B-�yB�pB>!B$=�B$B?BHB��B�mB)BF B�~BеB��BE1BK~B�$A�9�B:�B�DA�"�Ae��A��	@���@�)A�(A���A�l�AaU�AنRAE�1A��A6��AƘjA��AL%�A�r�B2>A��c@v�T@yOlA���C�N�A���A�iC���=J	�A13&A�d�Al��Af�AY|�AiQzAO~.ABPA̅MC���A�A�E�A�mA:��A3�}@ؤ�@���@	�A�z�A�+�B Ѵ?�LAǐ�Ad��A�6@�<C@��A�nA��]A�)�Ab>Aو�AF?A�{]A8� AƀIA��rAL��A�RbB�A�|I@g��@�&A��C�H�A��A��C���><�A0��A��AmK�A�tAYAuAi�APs�AA
�A�z/C��(A��A�~�A�QA:�fA5��@��@�\@��A�P�A�|�B ��?��            (   \   q      -   "                 
      d   6          	      M   E   >   !   4         
                  %   ,         
         	   E   x            
            5   ?   /      1   %   ;                     -   ;      +            5   3      )      !                                             !   '                        !      !      !                           !   -      '            !   +            !                                                            N��N�i�OY�O�ĹO^O���N��O��O��6Or�ROit�N�WfOKGN��vO
z�O6�LO���P5�N���O�GPN�R�N�e0O9)O�Z�P�'N�1ZO-�jN��O��gO��O��O~TN�<�N�N!'�O�&UOe�N��N\wTN���NS>N�0NLZ@O-s'O�J�N���N�O�N�-N	��  �  �  �  o  �  
�  .  �  *  �  �    V  ;    "  	X  �  �  R       �  �  �  �    �  n  �  �    �  �  �  �  �    �  �  �  �  �  
  �  �  �  �  %�t���o:�o<�9X=y�#=e`B<�C�=o<�j>@�<�9X<�j<�j<ě�<�j<ě�=�7L=49X=t�=C�=\)=�P=aG�=aG�=@�=�w=e`B=49X='�=<j=<j=@�=L��=@�=@�=P�`=Y�=aG�=m�h=q��=q��=u=}�=� �=Ƨ�=�\)=��P=� �=\ZX[]gktv������tsg`[Z���)02)�������315@BFN[]adc][NDB:53�����������������YXYYXZ`ht�������th[Y����)/7=;5)����RQY[git���~xtg^[RRRR������������������)26:=96)��?;:;?BLN[gryzxtg[NB?����������������������������������������������������������������


���������������
!)*+)��!)+46BCORUXYZYOB/)!������
 $$!
���������)5BMPOSQL>)�5007<DHTQHF<55555555 ��)DNNSTPB6)hipt}����������thhhh�������������������� �
#/29<<2/#
#"'5N[gt}�����tgB5)#�����
#/7IJG/#
����������� ���������������������������������������������������������������7COTZXOC6*(*777������"*&$�������� 
 ## 
�����
##$#
	������

���������

�������������������������
 ## 
�����������������������/26BEO[d[[OB:6//////����	������������������������55<@HJUVYUTHD<555555�����������������������������������������������!&&#������ 	������"&/;>B@;/"

z{��������zzzzzzzzzz�a�n�t�zÆÇÈÇ��z�n�a�U�U�H�H�O�U�^�a�G�T�V�V�W�Y�X�T�G�@�<�;�?�B�G�G�G�G�G�G�{ŇőŠšŠŞŖŔŇ�{�n�b�_�W�b�d�n�u�{���������ûܻ�������׻��������������'�4�@�M�Y�f�r�}�}�r�f�Y�M�@�4�$���"�'�
��#�4�=�;�5�0�#�
�������������������
������������������������������������������(�-�;�;�-������ۿ̿ǿƿѿݿ�����"�.�;�G�T�^�a�]�R�G�;�.�"�����
��"�6�B�O�[�h�l�t�u�w�r�h�[�O�B�3�*�)�+�3�6�s���������������������s�f�e�c�d�f�n�s����%�(�0�-�(��������������(�4�A�G�I�H�A�:�4�(�����������(�a�n�r�q�r�n�a�U�U�J�U�U�a�a�a�a�a�a�a�a������������������������������������Ѿ��������ʾ;׾׾Ҿʾ��������������������[�g�t�x�g�[�N�B�1�)�'�,�5�B�N�[Ƨ�����������$�$�������ƳƚƁ�t�vƃƧ�����������������������������������޻:�F�_�x���v�_�W�:�-�!�����������:�-�:�F�O�S�T�S�R�F�>�:�-�)�'�%�%�-�-�-�-�5�A�B�N�Q�W�X�Z�N�G�A�5�5�*�(�'�(�2�5�5D�D�EEEEE EEEED�D�D�D�D�D�D�D�D��	��"�2�D�E�A�9�/�"������������������	������/�;�a�m�u�s�f�H��	��������������DbDoD{D�D�D�D�D�D�D�D�D{DoDlDbDaDbDbDbDb���������ùϹٹܹ�ܹڹϹù��������������������(�*�+�(�������߽۽ݽ���������������������ŹŭŢŋŉŔŠų�����ҿ��������y�m�f�d�e�f�k�m�y���������������y�����������������������y�l�j�f�`�`�l�y�׾����"�-�.�"���	������׾Ӿ;׿m�y�z��y�w�m�a�`�T�M�G�C�G�S�T�`�d�m�m�������ʾ׾ھ׾ʾ������������������������Z�f�s�������s�g�f�Z�W�Z�Z�Z�Z�Z�Z�Z�Zù��������������àÓÇ�z�w�v�v�zÇÓàùE�E�E�E�E�E�E�E�E�E�E�E�E�EuEjEiEeEiEuE��������������������������������������������
�������
�
�����������������������
�
��
����������������������������4�A�M�O�N�M�A�6�4�-�4�4�4�4�4�4�4�4�4�4��(�,�(��������
���������@�J�M�Y�d�f�r�f�Y�N�M�@�@�=�@�@�@�@�@�@�����ʼ˼ּ׼޼�ټּʼ������������������r�~�����������������������~�r�\�R�T�b�r�	���"�&�'�"��	���������������	�	�	�	������
���
���������������������������\�O�C�6�*�&�*�0�6�C�O�\�c�\�\�\�\�\�\�\�Y�e�q�r�z�r�e�Y�X�U�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y   ( g F  3 ,   0 * 2 9 6 7  A ( l C U $ 0 ? R C L Q [ ! j L a ] 4 ! c � p = � w ;  6 % N F      [  U  d  �  �  �    @  �  �  �  T  �  ;  �  i  2  �  q  !    �  $  �    �  !  �  D    �  �  W  \  �  G  7  �  �  :  v  �  }  7  �  �  �  4  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  At  t  �  �  �  �  �  �  �  �  �  �  �  �  �  V  (  �  �  �  _  �  �  �  �  �  �  �  {  m  d  Z  Q  D  5  &       �   �   �  �  �  �  �  �  �  �  �  ~  o  _  Y  T  H  0    �  �  /  �    $  .  6  >  I  `  n  d  G    �  �  	  �  �  r  H  	  Q  �  �  �  �  �  i  |  �  P  �  �  �  �  ,  �      e  _  ~  �  	G  	�  
6  
~  
�  
�  
�  
�  
�  
�  
e  
  	�  	  x  �  �  �  3  �       +  +  !      �  �  �  �  u  N    �  b  �    �    H  g  �  �  �  �  �  �  �  �  �  �  \    �  (  �  �  e  �  �  �       *  %      �  �  �  �  f  6  �  �  B  �  G  �  #    �    �  .  �  �    �  �  �  �  1    K  �  
�  �  �  �  �  �  �  �  �  �  �  �  k  C    �  �  h    �  @  �  �  �  �  �              	  �  �  �  �  |  :  �  �  R     9  K  U  V  V  T  R  L  D  5  "    �  �  �  N    �  	  �    )  5  9  :  :  8  4  /  .  (  �  �  �  6  �  �  -  �      �  �  �  �  �  �  �  �  �  m  W  @  (    �  �  @  �  "            �  �  �  �  �  �  �  ]  1  �  �  |  <   �  +  }  �  �  �  �  	=  	W  	C  	  �  �  E  �  �  b  �  �  �  �  �  �  �  �  �  �  �  }  O  +  	  �  �  \  %  �  �    b  �    ^  �  �  �  �    t  d  O  5    �  �  �  �  M    �  �  M  N  ?  %  �  �  �  G  �  �  �  �  T  X    �  s    �  +            �  �  �  �  �  �  �  �  	  !  #    P  �  q           �  �  �  �  �  p  Z  D  6  &    �  �  �  n  v    D  w  �  �  �  �  o  D    �  �     
�  
  	2  6  %  �  o  �  �  *  ^  �  �  �  c  0  �  �  �  �  �  H  �  X  �  �  �  a  �  �  �  �  �  h  `  e  \  %  �  �  +  �  ]  �  %  e  �  �  �  �  �  l  =  	  
�  
�  
F  	�  	�  	_  	  �  N  �    w  �  F  �  �  �  �  �  	    �  �  :  �  �  #  �    y  �  a   �  A  �  �  �  �  �  �  �  d  5    �  �  3  �  x    �  0  �  n  d  T  ?  $    �  �  �  v  J    �  �  s  :  �  _  K  �  �  �  �  �  �  t  a  K  3    	  �  �  �  �  �  �  i  8    �  �  �  �  �  p  ]  F  -    �  �  �  �  `  2  �  �  2   �    �  �  �  �  o  I  "  �  �  �  �  h  M    �  �  g     �  �  �  �  �  �  �  r  U  3    �  �  w  D    �  �  �  {  1  �  �  �  �  �  �  �  �  �  {  q  g  ]  Q  A  1           �  �  �  �  �  �  �  y  b  J  1    �  �  �  �  y  V  2     �  �  �  �  �  e  B     �  �  �  �  N    �  �  e  <    �  �  s  �  �  �  n  X  B  1  !    �  �  �  I  �  =  t  l  F          �  �  �  �  �  �  �  p  Y  C  /      �  �  �  �  �  �  �  �  �  �  �  �  �  t  c  R  B  4  %    �  �  �  e  �  �  y  d  P  :  $    �  �  �  �  �  `  J  /    �  �  e  �  �  �  �  �  �  �  �  k  M  -  	  �  �  d  %  �  �  _    �  �    w  o  g  _  ^  a  c  e  h  j  h  \  Q  F  :  /  $  �  �  �  �  �  �  �  �  �  �  �  �  j  <    �  �  �  �  �  	l  	�  	y  	�  	�  	�  	�  
  
  	�  	�  	�  	x  	&  �    E  i  e  �  �  �  -  �  �  �  �  �  �  �  F  �  �    e  
z  	\    �    �  �  �  q  _  M  ;  +      �  �  �  �  �  �  �  y  g  U  �  �  v  b  K  3  &          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  i  d  V  C  4  %      �  �  �  %      �  �  �  �  �  �  r  ]  G  0    �  �  �  �  O  