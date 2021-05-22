CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Ǡ   max       P���        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =���        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�\(�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vc�z�H     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ǻ        max       @���            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >��        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�5   max       B3�>        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�E�   max       B4�        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?<Z)   max       C��#        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?M�)   max       C��-        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          A        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          %        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Ǡ   max       O�2`        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-V�   max       ?�����t        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >�        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�\(�     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vc��Q�     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ǻ        max       @��             \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Ca   max         Ca        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?z��vȴ:   max       ?���Q�     �  ^�                  
      	               
            7            |   	                  R         �   #      	                        #   $      F   	   
         0   	            
      C                     B   
               %      O���O�|^NI��N1ݫN`�O��XOP¸N�8N���Na��N,6AOC�jN�bN��O�N'E,P"�}O��N�m�O�ԡP��[M�{�NkO��O8||O�N+>�P��N�O	��P���Ol?�Olx�Oc"ZO ŒN���OxgO�Y�M�[�N40N��jO��QO�JoNԛ�P7�sO$��NͶ�N�O�N�A{O�N!N|��M��qN)�N.!O��N�>O��O!0ODZO_�QO`�1N`�pN�A�O�r!N���N��OL�mN��@N��OiR�NF��M�Ǡ�ě����㼓t��#�
���
��o:�o:�o;o;�o;ě�;�`B;�`B;�`B<t�<t�<#�
<T��<e`B<u<�C�<�t�<���<�1<�1<�1<�9X<�j<���<���<���<���<�/<�/<�/<�h<�h<�<�<�<�=+=+=C�=C�=\)=�P=�w='�='�='�=,1=,1=,1=49X=8Q�=@�=@�=L��=P�`=P�`=aG�=m�h=}�=�o=�7L=�C�=��P=��T=��=�j=�����������������������-.25BN[gt���vtgaNB5-������������������������������������������������������������X^WXgt�����������t[X �	"/9;>B;9/"	#+/7:92/#��������������������BBCHO[\`^[OBBBBBBBBB")-5:53)").5:>BCDB<5)��������������������&#$&*6ACOPRROGC96*&&�������������������������������������JJGFIN[ht�������h[OJ������
!$"#!
����<8BBO[\f^[OB<<<<<<<<#/4<FMRURH</'# ��������
����������������������������������������������zz�����������������z������ ����")06BMP[chmmh[OB62)"������������������������������������������������������������/)$$*/<GHPUSNH</////��������6SYK6�������������������������3,)*05BN[[hprqg[NB53GFN[gt�������tg][QNGzuu�������������zzzz#*++)#+(('+-/1<HPQMHG@<1/+mbbekx������������zm��������������������?BN[giga[NLB????????��������������������chnrvz�����������znc������!!����.(*//<HKU\YUHH</....!$6Yht��~~tgwp[B5)!llt��������������yul���������������������������

��������������������������������������������������������<<HMUWUUPHE<<<<<<<<<okpqt}���|xtoooooooonlklnuw{{~{znnnnnnn#/2:;77:/#!
#0:70*#
}|�����������������}��������������������
#0<CB50+#
!6BOchjhe[WOB62)'#!�����������������+.057BEGEB;5++++++++��������������������������������$!#)5BJDB<5)$$$$$$$$mghjmmzz}znmmmmmmmmm������������������������
"#&#
����#,/&#
��
#######�����"%%&&���


����











���������ûɻͻɻû��������v�l�g�l�s�x���U�n�zÀÄÂÃÂÄÃ�z�n�a�Y�O�H�=�=�G�U���������������������������������������ѹ���	������������������������������y���������������y�m�h�j�m�y�y�y�y�y�y�y�������6�B�?�1�0�6�C�6���� � �������H�T�a�m�w�z�������z�u�m�a�Y�T�H�B�?�F�H�z�����������������������z�w�u�v�z�z�z�z����)�6�=�6�3�+�)������������y�|�������������y�r�m�x�y�y�y�y�y�y�y�y�Z�b�f�h�f�`�Z�O�M�G�M�S�Z�Z�Z�Z�Z�Z�Z�Z���ʾ׾���������׾;ʾ����������������'�3�8�<�<�3�'������������`�m�y�����������{�y�m�`�Z�T�R�T�U�\�`�`�����������������������������������	���"�-�,�"����	��	�	�	�	�	�	�	�	�M�s�����ʾ׾���׾ʾ��s�f�S�J�A�?�>�M���5�B�K�[�@�5�)�������������������ѿݿ�����ݿѿпʿ˿ѿѿѿѿѿѿѿѿT�m�y�~�}�y�m�`�T�G�;�.�!���"�&�.�H�TÇàó��������������ùàÇ�H�=�H�R�Y�aÇ�n�o�zÇÈÇ�{�z�n�m�m�m�n�n�n�n�n�n�n�n������������������Ƽ�������������������̾����������ľ�������������w�s�t���������_�l�x���������������x�_�S�F�7�:�@�F�W�_�#�%�/�2�8�<�<�6�/�*�#�������� �#������������������������������������������(�5�N�_�t���s�g�^�Z�N�A�(����������������������������������������������D�D�D�D�EEEEED�D�D�D�D�D�D�D�D�D�DӼM�f���ȼ��������ּ��������z�f�Y�9�6�M���(�A�M�Z�e�f�k�s�|�s�f�A�(������B�O�[�h�t�|��|�u�t�h�[�O�B�:�;�:�;�@�B�������!���������ܽڽݽ޽�����)�6�=�B�J�J�D�B�6�)�(�����������(�3�5�@�5�(���
����������N�Z�g�s�����������������s�h�g�Z�U�N�J�N�[�h�tāčĚġĥġĚėčā�t�l�h�f�e�[�[�������������������������������������������������������y�w�v�y���������������������(�,�.�(�$��������������������6�A�<�9�)��������������������	�"�;�G�T�b�a�Y�G�>�"����������	���������	�������������������������޻л���'�@�V�X�@�4����û������������оʾ׾������������׾Ѿɾ����������ʼּ��������ݼּʼ����������ʼμּ־��������ʾ׾ؾ׾Ӿʾ����������������������������������ĽʽĽ½����������������������������������������������������������	���"�%�&�"��	����������	�	�	�	�	�	ÓÔÞàéàÓÓÇÄÇÓÓÓÓÓÓÓÓÓ���������������������������������������������ĽнؽнĽ��������������������������M�Z�^�f�s��v�s�Z�M�A�4�(�!�&�(�4�A�M�M�f�s�������������������z�s�m�f�`�]�f�f�s�������������������������s�T�A�=�F�f�s���$�(�+�/�2�2�(�(������������	��������!�&�+�)�!������߼���ﺤ���ɺȺ����������������x�q�r�z�~�������������������������y�l�f�_�_�\�`�l�y�|���=�I�V�a�b�j�b�V�I�B�=�4�=�=�=�=�=�=�=�=ù��������������ùìåêìòùùùùùù²¿�����������������¿°®¯¬§²���	��� ���	�����������������������Ź����������������żŹŹŹŹŹŹŹŹŹŹ����������������������������������������E�E�E�E�E�E�E�E�E�E�EEuEoEpEtEuE{EE�E�ǭǡǔǈǅǇǈǑǔǡǭǶǴǭǭǭǭǭǭǭ�#�<�I�Q�U�_�_�U�P�<�0�#��
�������#�H�<�0�#�����#�0�<�@�I�K�H�H�H�H�H�H�3�/�3�@�F�L�N�L�@�3�3�3�3�3�3�3�3�3�3�3 $ N S L < Q !  > 1 < . 6 - 1 l 7 P E E " | 8 , B Z E ; k  ^ N 3 + : 6 W @ U d ? 9 2 . Q _ G S D $ * Z b � b T b # ( = L ^ K 0 0 S J k I A l I    6  u  v  n  �  �    �  p  N  �  �  �  Y  p    �  �  0  >  c  x    �  e  Q  �  c  )  �  =  �  �  %  �  Z  T  +  �  �  �  �  �  e  �  �    �    �  1  H  b  l    �  d  �  �  �  �  �  �  �  I  �  N  �  �  �   �;�`B;�o�u�ě��D��;��
<�/<#�
<�o<���<#�
<�9X<�C�<T��<�j<T��=��=#�
<�t�='�>\)<�/<���=C�=<j=\)<���=�
=<�`B=T��>��=u=e`B=\)=8Q�=�P=8Q�=u=o=+=�P=�7L=�O�=Y�=���=0 �=@�=L��=Y�=�9X=L��=8Q�=<j=<j=Y�=T��=�x�=��=�+=�\)=�\)=ix�=��->o=��P=�\)=�1=���=\=��=ȴ9=���B"l�B�4B�\B �B,�B
J�A�5B�aB��B�pB��B��B�
B0f,Bp�B��B Bp�B�)BS�B;B��Bc�B�B}�B�B\3B_TB.�B�B�]Bt�B/�B	s�B
�%B��B��B ��B�Bk�B+�8BFZB�KB�B�,B3�>B!�FBn�B!��B@7B�ABK�BO�B(�CBB$�B�=BW�B%7}B@7B,��BWCB!ÎB�nB(�A��lB�%B�WB5�B��B,�B$;B"A�BABÎB aB,�BB
l�A�E�B�|B��B�jBΊB��B��B0{2B��BӎB=nBA�B@�B��BA	B�B@&BObB��B̣BG�B@�B@%B��B=BGIB��B	�B=^B�~B��B ��B�\B��B+�sB�rB��B�FB@zB4�B"�B@HB!�0B2B�B?�B@�B(PBF=B$��B?�B�B%?�B�B,�#B�rB!��B�B=A���B,�B�B@jBG�B-B$"G@��SA��oA��?<Z)Ams�A�A%A��A�-<A��^A��A>��AR!�?���Akz�A���A��RAH�A��A|��Af2CA�?^A���B{AI��@��A�]�AuEIA��A�g�C�9e@� �A:��A�U�A0AA���A��A��A�1�A�B&Ao?A���A�E�A`
A��n@�x�AS�WAjAM��A"	�A�v�A���A�Ȏ@�a�A%aA=9eADN�A�|A���A��@��A$�Bv+A��)A�{�A���A��+A�k,C��#B�;A�2A�x<?��_@��`A�u[A�r_?M�)Am�GA���A�||A���A�oA�@A>�nARԝ?�o�Ak
qA��oA�j�AG�$A��AA} Ad�'A�2�AȀB\�AJ�]@��A��Au A���A��JC�86@�)A;�Aڂ\A0��A�x�A��9A�u/A��hA�kXAo�A���Aԗ�A`��A�w,@��|ATy@�'AM�A!��A��A�EnA�w$@퐓A'��A=��AD�A�$A�}�AB�@��A B�A͑ZA��A���A���A�x[C��-B��A�uA���?��                   
      
                           8            |   
                  S         �   #      	      	                  #   %      G   	            0   
            
      D                     C                  %                        #                                 +   '         5                     )         A                                          5                                    '                                             
                                                   #                                          !                                          %                                                                                 
N���O�|^NI��N1ݫN`�OX�N�1�N��N/@ Na��N,6AO"a�N�bN��O�]N'E,OϗO��N�m�O}��O��M�{�NkOgXYN���N�Y N+>�OU1CN�O	��O�l$O!}6Olx�Oc"ZO ŒN���OxgOi�CM�[�N40N(��O��QO���No��O�2`O$��NͶ�N�O�N�A{OkH�N|��M��qN)�N.!O��N�>O�:�OM4ODZO_�QN��hN`�pN(�O�e�N���N��OL�mN��LN��OiR�NF��M�Ǡ  �  B  R  -  d  3  �  �  �  L  m  �  h  �  I  �    �  c  �  *  �  �  �  �  �  �  
  O  ^  E  �  
  A    l  7  �  �  .  �  {  *    �  ;  {  �  L  *    H  �    �  �  
  �  �  m    w  �  
F  �  �  �  I  .  	  y  ӻ�`B���㼓t��#�
���
�D��<T��;�o;ě�;�o;ě�<t�;�`B;�`B<#�
<t�<���<���<e`B<�C�=���<�t�<���<�9X<�`B<���<�9X=m�h<���<���>�=+<�/<�/<�/<�h<�h=C�<�<�=o=+=��=#�
=L��=\)=�P=�w='�=Y�='�=,1=,1=,1=49X=8Q�=}�=D��=L��=P�`=q��=aG�=�%=�7L=�o=�7L=�C�=���=��T=��=�j=�����������������������-.25BN[gt���vtgaNB5-������������������������������������������������������������YZgt������������tg`Y	"/35/,"	#/064/)#��������������������BBCHO[\`^[OBBBBBBBBB")-5:53)")+57<@?75)$��������������������&#$&*6ACOPRROGC96*&&��������� ����������������������������MMOS[ht�������th[WQM�������

����<8BBO[\f^[OB<<<<<<<<#/3;ELQSQH</#"�����������������������������������������������������������}}�����������������}����������/69BO[\hhhe[OB;6////������������������������������������������������������������/)$$*/<GHPUSNH</////��������������������������������3,)*05BN[[hprqg[NB53GFN[gt�������tg][QNGzuu�������������zzzz#*++)#+(('+-/1<HPQMHG@<1/+iipz}������������zmi��������������������?BN[giga[NLB????????��������������������chnrvz�����������znc����������-//9<@HQOH@<7/------('&,3?Uht�{smhe[O6,(llt��������������yul���������������������������

��������������������������������������������������������<<HMUWUUPHE<<<<<<<<<okpqt}���|xtoooooooonlklnuw{{~{znnnnnnn#/2:;77:/#!
#0:70*#
����������������������������������������
#0<CB50+#
!6BOchjhe[WOB62)'#!��������������������+.057BEGEB;5++++++++��������������������������������$!#)5BJDB<5)$$$$$$$$mghjmmzz}znmmmmmmmmm������������������������

"#%#
����#,/&#
��
#######�����"%%&&���


����











�����������������������������������������U�n�zÀÄÂÃÂÄÃ�z�n�a�Y�O�H�=�=�G�U���������������������������������������ѹ���	������������������������������y���������������y�m�h�j�m�y�y�y�y�y�y�y���2�>�<�/�.�6�=�6��������������a�e�m�r�u�s�m�a�V�T�O�L�T�X�a�a�a�a�a�a�������������������}�z�y�z�}���������������)�6�7�6�*�)����	���������y�|�������������y�r�m�x�y�y�y�y�y�y�y�y�Z�b�f�h�f�`�Z�O�M�G�M�S�Z�Z�Z�Z�Z�Z�Z�Z�׾��������׾ʾ��������������ʾξ׺��'�3�8�<�<�3�'������������`�m�y�����������{�y�m�`�Z�T�R�T�U�\�`�`������� ����������������������������	���"�-�,�"����	��	�	�	�	�	�	�	�	�������ʾԾپվľ�����s�f�Y�U�T�Z�f���������#�)�3�)�$��������������������ѿݿ�����ݿѿпʿ˿ѿѿѿѿѿѿѿѿT�m�y�}�|�y�m�`�T�G�;�.�$��!�,�.�;�L�T�zÇÓìù��������ùìàÓÇ�v�o�n�q�u�z�n�o�zÇÈÇ�{�z�n�m�m�m�n�n�n�n�n�n�n�n������������������Ƽ�������������������̾����������¾�������������z�v�v���������l�x�y���������x�p�l�d�_�S�G�K�S�_�`�l�l�#�,�/�3�4�0�/�#�#�!�������#�#�#�#�����������������������������������������(�5�A�N�\�_�Y�Q�N�A�5�(� ������#�(����������������������������������������D�D�D�D�EEEEED�D�D�D�D�D�D�D�D�D�DӼ����ʼټ���ݼԼʼ��������������������4�A�M�Z�^�b�e�Z�W�M�A�4�(�����(�.�4�B�O�[�h�t�|��|�u�t�h�[�O�B�:�;�:�;�@�B�������!���������ܽڽݽ޽�����)�6�=�B�J�J�D�B�6�)�(�����������(�3�5�@�5�(���
����������N�Z�g�s�����������������s�h�g�Z�U�N�J�N�tāčėĚĞģğĚēā�t�q�k�i�g�c�`�h�t�������������������������������������������������������y�w�v�y���������������������"�(�����������������������6�A�<�9�)��������������������"�.�;�G�O�]�\�T�G�;�.�"��	���������	�"���������� ��������������������������л�����'�4�B�B�4�'����ܻĻ����ûоʾ׾������������׾Ѿɾ����������ʼּ��������ݼּʼ����������ʼμּ־��������ʾ׾ؾ׾Ӿʾ����������������������������������ĽʽĽ½����������������������������������������������������������	���"�%�&�"��	����������	�	�	�	�	�	ÓÔÞàéàÓÓÇÄÇÓÓÓÓÓÓÓÓÓ���������������������������������������������ĽнؽнĽ��������������������������M�Z�^�f�s��v�s�Z�M�A�4�(�!�&�(�4�A�M�M�f�s�������������������z�s�m�f�`�]�f�f�s���������������������������s�j�g�X�b�s����(�(�1�1�(�&���������������������!�&�+�)�!������߼���ﺤ���ɺȺ����������������x�q�r�z�~�������y�����������������y�n�n�m�x�y�y�y�y�y�y�=�I�V�a�b�j�b�V�I�B�=�4�=�=�=�=�=�=�=�=ù��������ýùìéìîøùùùùùùùù¿������������������¿´²³²º¿���	��� ���	�����������������������Ź����������������żŹŹŹŹŹŹŹŹŹŹ����������������������������������������E�E�E�E�E�E�E�E�E�E�E|EuEqEpEtEuE{EE�E�ǭǡǔǈǅǇǈǑǔǡǭǶǴǭǭǭǭǭǭǭ�#�<�I�Q�U�_�_�U�P�<�0�#��
�������#�H�<�0�#�����#�0�<�@�I�K�H�H�H�H�H�H�3�/�3�@�F�L�N�L�@�3�3�3�3�3�3�3�3�3�3�3  N S L < R 5  P 1 < / 6 - ' l : C E F  | 8 & & _ E  k  # < 3 + : 6 W = U d ) 9 2 ' W _ G S D  * Z b � b T S ! ( = I ^ S - 0 S J j I A l I  �  6  u  v  n  '  �  �  W  p  N  Y  �  �  $  p  �  X  �     y  c  x  �  �    Q  �  c  )  �  \  �  �  %  �  Z  �  +  �  >  �    x  (  �  �    �  �  �  1  H  b  l    l  ;  �  �  �  �  E  �  �  I  �  3  �  �  �   �  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca  Ca    l  �  �  �  �  �  �  �  �  �  �  �  d  3  �  �  �    �  B  5       �  �  �  �  �  j  ;    �  �  {  H    �    @  R  S  U  V  X  Y  [  V  O  G  @  8  1  )  !        �  �  -  *  '  $  !            �  �  �  �  �  �  �  �  �  �  d  `  [  V  Q  M  H  C  >  :  3  *  !         �   �   �   �  2  2  3  -  '  &  &      �  �  �  �  �  �  �  i  I  >  2  [  �  �     H  d  w  �  �  �  �  q  R  ,  �  b  �    l  �  �  �  �  �  �  �  �  �  �  �  }  e  M  5       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  [  >       �  �  L  ?  1       �  �  �  �  \  +  �  �  �  P    �  �  �  S  m  g  b  \  V  O  B  6  )      �  �  �  �  �  �  �  h  N  �  �  �  �  �  �  �  �  �  �  g  H  &    �  �  �  O    �  h  \  P  C  5  &      �  �  �  �  �  v  S  .  �  �  `   �  �  �  �  �  �  �  �  �    t  h  [  N  @  1  "       �   �  C  H  G  C  =  5  /  (  !    
  �  �  �  �  �  s  o  e  H  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  i  ]  Q  D  8  �  �  �  �          �  �  �  �  h  %  �  k  �  g  �    �  �  �  �  �  �  �  �  �  �  �  �  �  m  @    �  �  0  �  c  _  \  Y  V  R  N  J  F  B  A  C  E  G  I  M  Q  U  Y  \  y  �  ~  l  U  6    �  �  �  �  a  .  �  �  D  �  @    �  <  �  h  �  	`  	�  
F  
�  
�  (  $    
�  
�  	�  	    �  `  �  �  �  !      �  �  �  ~  [  8    �  �  �  }  V  /    �  �  v  h  Y  H  7  #    �  �  �  �  �  _  8    �  �  �  p  �  �  �  �  �  �  �  �  �  �  �  n  T  1    �  �  �  �  T    I  q  �  �  �  �  �  v  V  1  	  �  �  c  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  8    �  �  _  -  �  �  �  �  �  �  �  �  �  �  �  v  j  ^  Q  F  =  3  *         r  �  |  �  	E  	�  	�  	�  
  	�  	�  	�  	/  �  ,  �  �  �  �  �  O  G  ?  7  /  '          �  �  �  �  �  �  �  �  v  e  ^  9    �  �  �  c  2  �  �  �  Y    �  z     �    w  �  �  �  o  d  5    �  �  6  C    �    N  l  T        �  �  e  �  �  �  �  �  �  �  k  C    �  �  P  �  �  A  �  g  
    �  �  �  �  �  �  `  *  �  �  D  �  �    �  )  �  �  A  :  2  (        �  �  �  �  �  �  �  �  o  W  D  6  '      �  �  �  �  �  �  �  j  >    �  �  =  �  �  L  �  R  l  Y  F  3      �  �  �  �  �  �  k  U  ?  9  9  ;  C  J  7  $  	  �  �  �  �  �  x  \  :    �  �  �  Q    �  1  �  �  �  �  �  �  �  �  r  P  *    �  �  ]    �  Q  �  >  �  �  �  �  �  �  �  �  �  �  �  �  }  q  e  Z  N  B  6  +    .  $        �  �  �  �  �  �  �    k  Y  G  5  #     �  �  �  �  �  �  �  �  �  �  �  �  �  y  i  T  <  #   �   �   �  {  g  `  [  R  E  1    �  �  �  �  �  �  M    �  U  �  �    $  )  *  %      �  �  �  �  �  _  1  �  �    f  �   �  �  �  �  �            �  �  �  �  ]  ,  �  �  �  c  X  k  �  �  �  �  �  �  �  {  U  G  j  i  q  e  N    X  �  .   A  ;  2  )        �  �  �  �  �  �  �  �  �  f  F  %     �  {  t  l  Z  P  `  i  ]  Q  D  5  $       �  �  �  �  �  �  �  �  ~  l  Z  H  8  '      �  �  �  �  s  H    �  �  �  L  6  "    �  �  �  �  �  r  O  *     �  �  g  *  �  �  ]  �        '  *  $    �  �  �  �  N  �  �  -  �    6  K    �  �  �  �  �  �  �  �  {  e  M  6      �  �  �  �  �  H  O  V  ]  d  k  r  m  b  V  K  @  4  "    �  �  �  �  `  �  �  �  �  �  �  �  w  i  [  A    �  �  �  �  �  �  �  �                        
  �  �  �  �  �  �  �  w  �  �  y  e  R  =  '    �  �  �  �  �  �  u  P  .    �  �  �  �  {  s  l  e  ]  U  M  ?  1  !  
  �  �  �  �  y  J    	:  	�  	�  
  
  
  
  	�  	�  	�  	Y  	  �  h  �  G  �  �  C  �  }  �  �  �  �  �  }  g  O  2    �  �  s  <    �  �  �  �  �  �  �  �  �  �  �  �  e  E  #  �  �  �  {  K    �  �  T  m  i  d  X  I  6    �  �  �  V  u  V  >  6  (    �  �  �  �  �  �  �                �  �  �  �  n  0  �   �   a  w  r  m  h  d  _  Z  V  Q  L  K  M  O  Q  S  V  X  Z  \  ^  �  �  �  �  �  �  �  �  �  �  �    U  (  �  �  �  P    �  	�  
=  
E  
5  
  	�  	�  	�  	�  	[  	  �  V  �  ^  �  �  j  �   �  �  k  Q  4    �  �  �  �  x  S  .  	  �  �  �  �  q  [  D  �  �  �  �  �  �  �  �  �  �  �  �  u  k  b  Y  Q  H  ?  7  �  a  0    �  �  �  x  W  4    �  �  �  `  >  !     �  �  0  H  A  9  &    �  �  [    �  �  I    �  �  �  �  M  �  .    �  �  �  �  c  3    �  �  �  i  A    �  �  m    �  	  	  �  �  �  �  �  c  /  �  �  ;  �  v    �    �    �  y  j  \  M  ?  2  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  e  X  J  <  .         �  �  �  �