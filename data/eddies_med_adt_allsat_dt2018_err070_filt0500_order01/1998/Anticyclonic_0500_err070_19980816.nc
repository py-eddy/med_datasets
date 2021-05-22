CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N%њ   max       P��'      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =��`      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�(�\     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @v���Q�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @M�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @���          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >y�#      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�QQ   max       B-v      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�t�   max       B-��      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?6m�   max       C���      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =+�   max       C�k	      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N%њ   max       P��c      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����U�=   max       ?�� ѷ      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >C�      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E���R     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v���Q�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @M�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�V�          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E_   max         E_      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?�I�^6     p  S(               "   F   E      5         	   G      `      #   	               S         	   
   	         !   
   2            
      !   =         ^   '      	      �      
   3   )         /            &OGkN��N%њN�^FO�f�O�*�P��cNQ7�P�.@O� �ON�xOJ��PO��N�N&P��'N���P-' N��O��O��#O��N]SPF|�N2�ON��{N�ScN�(�N���N�MN@#Oz��N�)FP��<N(��NP(AN�@N��wNNfOf��P3D�O�,N\S�Oң�O�{�ONyN��-N&�.O���O�Na@�Of��Ow�N�L�N�j�OuPN��NA��Nh��O5=l�ě�����D���49X�#�
�D���o:�o:�o;�o;ě�<t�<t�<49X<�o<�o<�C�<�C�<�C�<��
<�1<�9X<�j<�/<�h<�<�=+=+=C�=C�=C�=0 �=0 �=0 �=<j=<j=H�9=H�9=Y�=ix�=q��=}�=�%=�o=��=��=�\)=�t�=���=��w=���=�1=� �=�j=�v�=ě�=ȴ9=��`pqx���������������tpd_ghkjpt~����|xtmgddC>GHIUX]UHCCCCCCCCCC����������������������������
������gdcgt������������tmg����)5FOVTMB5)��� 	".#"	          �����#<UeiRF93$
�����������������������)/6;AB<61)���������� ������������
/:JUfiaH/#
���-./9<EHUUZUPH<4/----�����BO]^OB5������������������������dbl}�������������tkdstw�������������ytss|zz|��������������������
#//+#
�����rsx���������������zreghtu�����utjgeeeeeeOm{~uskjd[OG0���������������������������������������� $).,)&    ##&0<;40#���������������������������������������������������������������������

�����"#.03<AILIG@<0-%#�����)@JOJB������# %)*56985,)########����� ������������������

���������������������������������������������������������������14BN[s��������tg[N81��������*-(�����aaUQHD<:<HUZabcaaaaa6BO[[WTKB6)
�������������������� $)6:BLLKB<6/)������������������������

������������������

�����`VWam������������zm`//31<HHHKIH<8///////ID<0# #0<IMMI��������������������%$)-5BDMKB5)%%%%%%���������������������������������������������������������#%&#"
)+2676)"����������	


������B�N�[�_�R�M�O�G�E�B�5�)����$�+�-�5�BŹ����������������������������žŹŵŹŹEPE\EiEiEiE_E\EPEGECEPEPEPEPEPEPEPEPEPEP����������������ܹڹչܹ޹��軞�����ûɻѻһ̻û��������x�m�x��������a�n�zÍâäàÖÇ�z�n�a�U�D�G�E�E�L�U�a�<�U�d�{ŘŚŗŋ�n�U�#�
�������������#�<�T�a�a�f�e�a�T�T�J�I�T�T�T�T�T�T�T�T�T�T�"�;�a�����������������a�T�;��	����"���(�A�Z�g���~�g�U�N�J�A�;�5��������������	����	��������������������׾����ʾ���������׾ʾ�������������������;�Y�y�������y�`�F�.�����������
�������������������������������������Ƨ�������������Ƨ�h�O�7�1�6�C�hƃƧ���$�)�6�=�=�;�6�)�'����������(�4�M�f���������s�f�Z�(��
���	���(�����������
���
����������������������ܻ�������$��������ܻл˻лѻܾ4�A�Z�\�o�r�r�j�g�M�A�4�*�'�(�0�/�3�-�4������&�/�6�4�(������нĽ����ɽݽ��{ǈǈǉǈ�|�{�o�b�a�b�b�o�u�{�{�{�{�{�{���׿	����׾ʾ���������v�r�����������!�-�2�-�-�!����
����������;�H�T�]�a�g�d�a�T�H�;�;�6�9�;�;�;�;�;�;�����������������������~�z����������������������������ּѼּ̼ݼ�����㿒���������ĿƿĿÿ����������������������M�Z�[�f�g�h�f�]�Z�M�A�4�.�(�3�4�A�K�M�M�
����#�&�#���
���������� �
�
�
�
�r�����������������������w�r�f�_�Y�\�r������������������������r�o�l�r��������Z�h�\�X�X�e�k�Z�A�(����ʿѿݿ���	�(�Z�����ĿǿĿ������������������������������T�`�h�h�`�T�G�?�G�I�T�T�T�T�T�T�T�T�T�T�(�5�A�B�M�N�Z�a�Z�N�5�(�������!�(ààìðùýùóìàÓÏÊÉÓßààààE�E�E�E�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�FF$F1FVFcFoFvF~FF|FoFcFJF=F1F)F!FFF�)�6�P�[�Y�B�;�:�6������������������)�S�`�l���������������y�l�`�S�O�K�J�K�M�S�������&�(�+�-�.�(�����������������Ƽɼ���������r�f�_�]�_�f�r����hāčĚĦĳļĽĳĦč�y�h�[�O�I�E�O�[�h�e�����������ǺɺϺ����������x�r�n�e�c�e�����������������������������������h�tāćąā�t�m�h�d�h�h�h�h�h�h�h�h�h�hD�D�D�D�D�D�D�D�D�D�D�D�D�D�DD{D�D�D�D�ŔŠŭŹ����������������ŹŲŪťŞřŗŔ¦²¸²¦¥�պɺź��������ɺֺ��������������o�{ǈǉǔǙǡǡǡǔǈ�{�o�b�a�_�b�c�m�o�	��"�/�0�/�,�%�"��	������	�	�	�	�	�	�
�	�����������
�������
�
�
�
�
�
�����ûлܻ���ܻлû������������������B�N�[�g�q�l�g�]�[�Y�N�J�B�;�B�B�B�B�B�B�U�b�n�{�|��{�n�b�b�U�R�U�U�U�U�U�U�U�U�_�l�o�l�e�b�_�S�H�F�E�@�F�J�S�\�_�_�_�_EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EtEnEiEu i u Y ' 0 8 $ a O d 6 K >  ? A # 6 L n 3 M 7 U % "  F ) Z , 1 F [ 7 [ + W U R p k  U w S B 3 p D * " ; ] ; F G q 4    �  �  .  	  8  1  �  �  \  �  �  �  s  �  �  �  �    I    �  x  j  M  �  �  �  �    j  �  �  �  G  V  S  �  �  ,  3  �  �  �  �    �  M  j  �  f  �  I  �  �  �  �  k  �  ���C��e`B��`B;o<�j=�+=�+;��
=]/<���<�C�<�C�=��
<�/=�S�<�9X=T��<���=��=49X=@�<�h=�
==C�=t�=�w=�w='�=Y�=t�=�7L=0 �=�v�=@�=H�9=ix�=e`B=y�#=��T=�l�=�9X=�7L>�w=��`=��=��=��w>y�#=ȴ9=�{>o=�F=Ƨ�=���>V=��=��`=���>\)B4rB	��B{�B��B#
VB
Y?B�MA�QQBj�B|B��B��B�B�B�B�BqMB:�B3�B#ȄBuPB	��B�5B &�B�aB}yB%s�B�B�sBU�B"�B%�B�B�B�BB!��Bv�Bn_B	�B-vB��Ba�Bv�B@�B�iB}�B�B L�BӹB%��B��B>�B�"B(�Bb�BV0B��B�QB��B	ýBA[B��B"��B
@�B�A�t�B�B:B4}B7|B��B�<BI�B�oB=�BF@B?�B$!�BJsB	��B��B ?�B�B��B%FxB?BB�0B��B"�aB&/�Bb�B��B>SB�2B!ʉBB�B�pB
� B-��B��B@	B��B@B��BC�B>9B>�B��B%��B�%B?JB̹B�|BC�BDgB�B�A��A��C��!?6m�@�A�Aǁ�A섳A��:A���A���A�{�AQaAfB"A��BhwA�`A<R�A�rO@���A;��A.�hB �AQ��@k}�A��iAq��AUUAt�,A<��A��q@��@�S�A��/Av	�AgeA�b�AˎC�v�C���A�h�Ap�A4"@�JbAݰ4@�.A�DhAܫ1C��tA�<�A�	b@B�RBh�A��[A�G�@���A�0A��@�ȪC�
^A��(A�4C�ǉ?J=�@��A�yFA�BA��iA�C�A���A��AN@6Ag��A� �B��AՆ�A;�EA�e@�|A=6A.��B��AR��@e�LA���Ar��AU4AuxA=OA���@��P@��!A��}AuAjAg�A�q�A��C�k	=+�AՆ.As2A5`@� �A߂�@�8A�\�A܃�C���A��LA��G@IVIB@XA��cA��@���A���A�u.@���C��               "   F   E      6         	   H      `      $   
               S         
   
   	         "   
   3                  !   =       	   _   (      	      �         3   )         0            '                     7      7   #         1      ?      )            #      1                              =                     -   !      !                                                                     -      +            %      3                        !                              9                     -                                                         OGkN��N%њN��
NѲ�O�APd�+NQ7�P�TO@
�ON�xOJ��O�g�N��WP�&�N���O�!�N�Y*O��O�ߠON]SO��QN2�ON��{N�ScN�(�N���N݃�N@#OD@qNm#P��cN(��NP(AN�@N��wNNfOf��P3D�Os��N\S�O�;kO��MONyN��-N&�.O"2O�Na@�Of��Ow�N�L�N�j�OuPN��NA��Nh��O5=l  �  `  �  �  �  M  �  �  �  �  �  �  7  l  �  %  �  �    7  �    �  �  l  �  �  �  �  R  �  �  �  �  �  C  f  �  �    �  G    V    �  F  �  �  �  	Q  K  �  �  	�    �    	��ě�����D���t�;ě�<�j<t�:�o<u<49X;ě�<t�=o<e`B=t�<�o<�<���<�C�<ě�=o<�9X=Y�<�/<�h<�<�=+=C�=C�=�w=t�=49X=0 �=0 �=<j=<j=H�9=H�9=Y�=�%=q��=��=��=�o=��=��>C�=�t�=���=��w=���=�1=� �=�j=�v�=ě�=ȴ9=��`pqx���������������tpd_ghkjpt~����|xtmgddC>GHIUX]UHCCCCCCCCCC���������������������������������������olmrt�����������wtoo�����)5DNOKB5)��� 	".#"	          ����
/HUWUC:/-#
�����������������������)/6;AB<61)���������� ���������
/<AQWZUH/#
�8/0;<=HRUUUJH<888888����)BNTVKC5)������������������������xx{����������������xw{�������������|zz|����������������������
#,+'#
���}{|����������������}eghtu�����utjgeeeeee)BKV]bc^XOB5 ���������������������������������������� $).,)&    ##&0<;40#�������������������������������������������������������������������� 
 ������� #/01<IIIE><0/(#    �����)?IOIB)������# %)*56985,)########����� ������������������

���������������������������������������������������������������14BN[s��������tg[N81�������%!�������aaUQHD<:<HUZabcaaaaa	)6BLPQOJB6)�������������������� $)6:BLLKB<6/)������������������������

������������������

������`VWam������������zm`//31<HHHKIH<8///////ID<0# #0<IMMI��������������������%$)-5BDMKB5)%%%%%%���������������������������������������������������������#%&#"
)+2676)"����������	


������B�N�[�_�R�M�O�G�E�B�5�)����$�+�-�5�BŹ����������������������������žŹŵŹŹEPE\EiEiEiE_E\EPEGECEPEPEPEPEPEPEPEPEPEP���������������ܹܹܹ����軞�������ûûû��������������������������a�n�zÇÉÎÇÇ�z�o�n�a�^�U�Q�O�U�X�a�a�<�U�b�n�{Ōŏŋŀ�b�<�#�
������������<�T�a�a�f�e�a�T�T�J�I�T�T�T�T�T�T�T�T�T�T�;�T�m�z�������������a�T�;�/����#�/�;��(�5�A�R�]�e�\�N�I�5�(�����������������	����	��������������������׾����ʾ���������׾ʾ������������������m�|�������y�j�`�;�.�"�����.�;�G�T�m���������� ��������������������������Ƨ�������������ƳƁ�_�R�L�R�hƁƎƧ���$�)�6�=�=�;�6�)�'����������4�A�M�f�s�}��v�f�Z�M�A�4�(�����(�4�����������
���
���������������������ػܻ�������$��������ܻл˻лѻܾ4�A�M�Z�`�f�j�o�o�e�^�M�A�4�/�*�+�0�3�4����������������ݽӽннݽ߽��{ǈǈǉǈ�|�{�o�b�a�b�b�o�u�{�{�{�{�{�{�ʾ׾����������׾ʾ��������������ʻ�!�-�2�-�-�!����
����������;�H�T�]�a�g�d�a�T�H�;�;�6�9�;�;�;�;�;�;�����������������������~�z����������������������������ּѼּ̼ݼ�����㿒���������ĿƿĿÿ����������������������A�M�Z�f�f�g�f�\�Z�M�A�4�1�*�4�5�A�A�A�A�
����#�&�#���
���������� �
�
�
�
�����������������������r�f�^�`�f�r�v���������������������r�q�p�r������������Z�f�[�W�W�d�j�Z�A�(�� �ݿѿ޿���
�(�Z�����ĿǿĿ������������������������������T�`�h�h�`�T�G�?�G�I�T�T�T�T�T�T�T�T�T�T�(�5�A�B�M�N�Z�a�Z�N�5�(�������!�(ààìðùýùóìàÓÏÊÉÓßààààE�E�E�E�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�FF$F1FVFcFoFvF~FF|FoFcFJF=F1F)F!FFF�)�6�P�[�Y�B�;�:�6������������������)�y���������������~�y�l�`�S�N�N�P�S�`�l�y�������&�(�+�-�.�(�����������������������������������r�h�c�e�m��hāčĚĦĳĻļĳĦč�{�h�[�O�M�H�O�[�h�e�����������ǺɺϺ����������x�r�n�e�c�e�����������������������������������h�tāćąā�t�m�h�d�h�h�h�h�h�h�h�h�h�hD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ŔŠŭŹ����������������ŹŲŪťŞřŗŔ¦²¸²¦¥�պɺź��������ɺֺ��������������o�{ǈǉǔǙǡǡǡǔǈ�{�o�b�a�_�b�c�m�o�	��"�/�0�/�,�%�"��	������	�	�	�	�	�	�
�	�����������
�������
�
�
�
�
�
�����ûлܻ���ܻлû������������������B�N�[�g�q�l�g�]�[�Y�N�J�B�;�B�B�B�B�B�B�U�b�n�{�|��{�n�b�b�U�R�U�U�U�U�U�U�U�U�_�l�o�l�e�b�_�S�H�F�E�@�F�J�S�\�_�_�_�_EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EtEnEiEu i u Y (  "  a U b 6 K X ( = A   ; L X  M " U % "  F , Z % = E [ 7 [ + W U R d k  U w S B $ p D * " ; ] ; F G q 4    �  �  .  �  �    �  �    �  �  �  "  �  n  �  g  �  I  /  /  x  �  M  �  �  �  �  �  j  �  �  �  G  V  S  �  �  ,  3  O  �  +  �    �  M  R  �  f  �  I  �  �  �  �  k  �  �  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  E_  �  �  �  �  �  �  �  �  �  �  �  �  w  j  S  /  
   �   �   �  `  ]  Z  W  U  Z  _  d  a  S  D  6  $    �  �  �  �  �  c  �  m  Y  E  1      �  �  �  �  �  �  l  S  ;        �   �  �  �  �  �  �  �  �  v  e  O  3    �  �  h  (  �  �  ^   �  ;  a  p  o  n  v  �  �  �  �  �  �  s  N  2    �  ]  c  7  	u  
,  
�  
�    9  H  M  F  +  
�  
�  
A  	�  	  ?    �  �  s  t  y  y  �  �  z  q  o  d  I    �  �  �  �  {    �      �  �  �  �  �  �  }  z  v  s  p  n  k  i  g  d  c  a  _  ]  �  �  @  r  �  �  �  �  S  (    R  u  j  N    �  ;  �    �  �  �  �  �  �  �  �  �  �  �  �  k  >    �  �  P    �  �  �  �  �  �  �  q  Y  A  (    �  �  �  �  �  e  P  0    �  �  �  �  �  z  g  T  @  +          �  �  �  �  �  �  �  �  
      /  7  -    �  �  �  f    �  :  �    7  �  E  U  a  i  k  f  X  @  )    �  �  �  �  �  c  D  %    �  z  �  �  �  �  �  �  �  �  p  .  �      �  �  .  Y  c    %  $  "  !               �  �  �  �  �  �  s  T  6    �  �  �  �  �  �  �  �  �  �  �  �  ~  W    �  m  	  �    [  n  �  �  �  �  �  {  j  Q  1  �  �  X    �  i  !  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  G  $  
    �  *  +      �  �  �  �  �  �  Z    �  t    �   �  �    5  R  f    �  �  �  �  �  �  q  H    �  �  V    e    �  �  �  �  �  �  �  �  i  I  (    �  �  �  o  G    �  �  �  &  V  �  �  �  �  �  �  {  F    �  b  �  P  �  c  �  �  �  �  }  g  P  <  )    �  �  �  �  �  w  [  ?    �  �  l  g  b  [  R  H  8  #    �  �  �  �  s  M  $   �   �   �   �  �  �  �  �  �  �  �  �  �  �  i  R  <  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  W  1    �  �  f  (  �  �  �  �  �  }  j  X  F  5  (      �  �  �  �  b  E  (  �  �  �  �  �  u  a  F  (    �  �  �  X  $  �  �  x  %  �  R  [  d  m  u  ~  �  �  �  �  �  �  �  �  �  �         .  �  �  �  �  �  �  �  s  W  :    �  �  �  R    �  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  7    �  �  S    �  �  h    �  <  �    �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  Q  -    �  �  �  �  �  �  �  �  o  \  K  ;  *      �  �  �  �  �  b  =    C  >  5  '    
  �  �  �  �  �  �  �  j  N  %  �  |  D    f  S  ?  )    �  �  �  �  �  i  A    �  �  �  s  K  �  l  �  �  �  �  �  {  j  Y  I  9  *      �  �  �  �  �  �  b  �  �  �  �  �  �  �  z  U    �  s    �  L  �  q  0  �      
�  
�  
I  
  	�  	Y  �  �  0  �  i  &    �  �  �  �  �  �  n  �  �  �  �  �  �  z  M    �  �  R    �  �  H    �  �  G  E  D  <  1  )  /  4  *      �  �  �  �  �  m  P  3    �  �  �  �    �  �  �  �  K    
�  
=  	�  	5  �  �  `     a  T  V  R  F  H  K  @    �  �  \  �  w  �  =  �  w    q  �    
  �  �  �  �  �  [  /    �  �  o  8    �  �  Z  �  +  �  �  �  �  �  �  �  {  v  q  k  b  Y  P  G  =  2    �  �  F  -    �  �  �  �  �  r  P  -  �  �  D  �  �  Q     �  [  �    {  �  K  �  �  �  �  }  C  �  (  !  �    $  �  i  W  �  c  >    �  �  �  }  >  �  �  U    �  y  W  �  R  �  =  �  m  =  �  �  f    �  �  `  #  �  �  �  {  _  :    �  �  	Q  	'  	   �  �  �  �  �  X  #  �  �  b    �  '  �  <  �  �  K  
�  
�  
�  
b  
%  	�  	�  	V  	  �  c  �  ~  �  �  �  �  R   �  �  �  �  �  �  �  x  e  P  8    �  �  �  �  �  r  #  �  5  �  �  �  l  G    �  �  �  b  ,  �  �  z  9  �  �  U      	�  	�  	�  	�  	~  	Y  	3  	
  �  �  Y    �  Q  �  (    �  �        �  �  �  �  �  �  �  n  V  >  )    �  �  �  i    �  �  �  �  �  m  U  =  &    �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	�  	�  	�  	|  	`  	4  �  �  s  &  �  p  	  �  %  �  �  �  �  <