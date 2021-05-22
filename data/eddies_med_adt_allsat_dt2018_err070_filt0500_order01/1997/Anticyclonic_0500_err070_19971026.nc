CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ǯz�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�7�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =���      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @F%�Q�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vs�z�H     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P            p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       >P�`      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�j�   max       B0i      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�^   max       B00      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?&��   max       C�}      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?/�   max       C�y      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�7�   max       P�\$      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?�����$      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =��#      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @F%�Q�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vs�z�H     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P            p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D9   max         D9      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?��Q�`     0  O�                                  c                  
   	   D         �   ~   
   ,      
         =         	   $   "      E   -                  8            	         '   H   A      	NN@�mN\�O�F�N_�:N��N�NUN*p�OE<<Pw�9O3�Ne�CM�7�N�P��N��N���P���Oh�O��eP���O�v:N�ئP�N��gNqփN̓�N�oFO��O�L�O�NȰ�O�KO��)N|dO�^P:{O��M��nN!��N�-Oa+3P��N�m�OW�O���N��$N`JcOgO7�^PۤO�_�N	AN{R�����P���T���D���D��%   %   ;D��;�o;�`B<t�<D��<D��<T��<T��<T��<�o<�t�<�t�<�t�<���<��
<�1<�h<�=o=o=+=C�=t�=�P=#�
='�='�='�=,1=0 �=0 �=0 �=8Q�=8Q�=L��=Y�=aG�=aG�=q��=}�=�o=�7L=��=���=���=���=���{��������{{{{{{{{{{���������������#0<>><0/##/<HUansnaUH/#��������������������346?BDOPOMIB<63333332556BNVSNCB522222222xz����������xxxxxxxx��������������������rsy��������������}tr������BJLFB5)����������������������������������������������������������������MORTabfa]TMMMMMMMMMMmmvu��������������m��������������������US[\gtx{ztigf[UUUUUU����	5NVXTN1)��������������JHKS\ht������yth[PJ����)BTdg`mnf[B5�����
"#!
��������		"/;B>;//"	������.0/(�������)55954))-6BEBB6)"#))))))	
#%(&$#
�������������������������������������������������������������������������������)*/6CIMIC6*�������#)26/#
����858::<GUaenpqnjaUH<8QKMUbnrnjebUQQQQQQQQ�������
"''"���������)5BGF=5������������
��������!����������������������������������������T[\_emz������zmaTOJT����)6O[huq[O6)��xz�������������zxxxx
#(*,,+)#
P\g���������������gP������������������������

������������~||}�����������������������

���tpx������ �����ztz�������������|��{z)11,)&')*1-)�zÇÈÍÇ�|�z�y�r�o�z�z�z�z�z�z�z�z�z�z�Ľнݽ����ݽнĽ½½ĽĽĽĽĽĽĽĻ����������������������������������������������������� ����������������������� �����������������������������ʼּ׼ּּּؼּмʼ������ƼʼʼʼʼʼʿG�T�T�T�M�P�G�;�8�;�=�@�G�G�G�G�G�G�G�G���������������������������������������ݽ����ݽнĽýĽнؽݽݽݽݽݽݽݽݹܹ�������
�����ܹԹѹϹɹϹ׹��0�I�U�b�n�p�n�j�[�U�I�
��������������0Ŀ������������������������ĿĳįħĳĹĿE�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�Eټ��#�'�(�'�%��������	������Ŀ��������������ĿĸĿĿĿĿĿĿĿĿĿĿ�M�Z�s�������������������s�Z�A�:�;�8�@�M�ûȻлٻܻһлû������������ûûûûûÿy���������������y�o�m�l�m�r�y�y�y�y�y�yƚƳ������������Ƨƒ�\�H�:�C�W�g�oƚ�.�;�G�T�`�m�y�{�y�j�`�T�G�;�.�&���"�.�A�M�Z�f�s�}��������s�f�Z�A�4�,�#�*�4�A���B�O�hĎĕĈ�t�[�N�6����������������D�D�D�D�D�D�D�D�D�D�D�DyDtD{D�D�D�D�D�D��T�a�a�m�n�s�x�x�m�a�]�T�T�N�H�F�H�J�T�T���������������������������a�V�f�p�z���������ĿοϿ̿ĿĿ������������������������@�3�.�.�'�������'�/�3�@�@�@�@�@�@�����������ʾоʾľ����������������������A�N�Z�a�g�l�g�`�Z�V�N�A�A�=�A�A�A�A�A�A�)�6�B�O�U�V�Q�O�B�)���������
���)�������������������������������v�u�w����àìùý��üùìàÓÇ�}�z�x�zÂÇÓÛà�m�y�����������������y�m�g�b�e�i�m�m�m�m�N�[�g�j�a�T�:�)������������������)�N�m�������������������������v�m�b�]�\�_�m��������������߼޼�����������	�"�4�;�F�H�K�H�;�/�"��	�������������#�<�@�8�/�%� ���
�������������������#�����������������������x�����������������N�Z�g�r�g�f�Z�N�L�N�N�N�N�N�N�N�N�N�N�N�Z�f�r�f�d�a�Z�X�M�K�M�R�Z�Z�Z�Z�Z�Z�Z�Z�-�:�F�I�F�:�5�:�F�R�U�F�:�-�"�!���!�-ŔŠŭŹ����������ŹŭŠŗŌņ�y�x�{ŇŔ�ܻ������������ܻл˻ӻ������û��
�����!�#�%�#��
������
�
�
�
�(�4�A�M�Y�f�q�s�}�s�f�Z�M�A�4�&����(���������������������x�s�`�c�p�����������y�����������������������{�y�s�x�y�y�y�y�G�J�G�E�>�;�.�"� �"�"�.�;�=�G�G�G�G�G�G��(�5�A�N�Y�W�N�K�A�5�(��������EuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEuElEmElEu���ֺ���պǺ��������������������������B�M�V�]�Y�M�'�������߻ллܻ���'�B�����������������������������������������g�t�x�t�n�g�f�c�g�g�g�g�g�g > . 6 @ F R g ^ O " * / C f J ( Q # < 3 J # B ` C ? W > < K > # ; � @ 4 8 L T S L Z \ X ~ U v H ` 4 / v A S j    N  a  y  O  �  :  _  �  A  �  	  ~  �  &  3  _  �  �  �  �  +  �  �  5  �  �  �  �  �  �  G  B  �  �  Q  �  #  �  d    ?    �  �  *  z  8  �  �  M  �  X    +  Žo�+����<����`B:�o;o;D��;��
<��=��<�<���<u<u='�<�9X<���=�{=��=@�>P�`>�<�=��=t�=,1=49X=�w=\=��=aG�=D��=���=��P=D��=�G�=�-=�\)=@�=L��=}�=��=�/=u=���=���=�hs=�O�=��T=�S�>�+>\)=��
=�^5B
��B��B%��B�AB�/B:=BȠB'uB�XB^/B�zB{Bv�B"-fA���B�TB"i�B	C�BE{B��B��BfBAxA�j�Bi�Bk�B� BH�B>B:�B B!�HB0iBQ}B��B'��B�YBd/B�VBs�B#�B [�A��BmB\�B$٨B
�B,kBO�B=�By�BՐB��B;B(B
��B��B%�B�(B��B9yB1[B�B��BABB@B�~B"-�A��PB��B"�VB	P�B�B��B�ZB@eB@A�^B3@BA�B��B@�B �fB�KB?�B"<�B00B�B��B'L[B�B��B�3B��B2�B XA�{�B��BN�B$��B�B,�B@�B�=B�:B��B��B77BC<Aȣ�A)�@���A�i|B�s@��yAe+A�8�A*|�?&��A�`A��C�}@Ü�A�_3ABذ@�j$AnmDB:EAeiA>�8A��4C��A�n�A�Awp�?���AK��A��A���A��A�V�Al��A��A��wA��A���A���A��A��)A?�@z�GA��a@�nA��-A;qA���AljAbJ�A�R�C�Y@*	�@�e�@i�A��eA�pA*�r@�&A�~~B�@�V}Ad�A�u�A*�C?/�A�U4A�n�C�y@��A���ABo�@�NAm|B�Ac��A=*�Aӳ1C��kA�zZA���Aw:?�رAL��A���A׀A���Aˆ�Am�A��oA���AXA��iA���A���A�wMA>�@|�A���@��A�bpA:��A��\A��Ab˼A��C�@*��@�t�@L�A��}            !                     d                  
   
   D         �      
   -               >         	   $   "      E   -                  8            
         '   I   B      	                                 /               %         9         9   !      +                  %         +         #   )                  +         '               +   %                                                               5                  +                           +            !                           '                        NN@�mN\�N�L�N_�:N��N�NUN*p�N���O�oOGINe�CM�7�N�OE��Nfk�N���P�\$Oh�OjKO���O4�DN�ئP�N��gNqփN�y8N�oFO`7�O�n!O	��NȰ�O�KOvn�NE��O���O�Q�O��M��nN!��N2�[Oa+3O�|JN�m�OQO���NH~�N`JcOgO.��O���O��~N	AN{R�    �      ]  �    �    �  	�  {  9  �  �  �  I  K  >  9  v  �    �  S  Z  �  F  �  
D  �  h  �    �  n  	�  d  �  ;  <  �  #  t  �  �  �  L  �  k  
c  
  	�    ֽ���P��:�o�D���D��%   %   ;D��<D��=@�<49X<D��<D��<T��<�/<e`B<�o<ě�<�t�<���=��#=�o<�1<�h<�=o=+=+=P�`='�=��=#�
='�=49X=,1=e`B=H�9=0 �=0 �=8Q�=T��=L��=�O�=aG�=ix�=q��=�o=�o=�7L=��P=ȴ9=�1=���=���{��������{{{{{{{{{{���������������#0<>><0/#%%(/<HNUVUNH</%%%%%%��������������������346?BDOPOMIB<63333332556BNVSNCB522222222xz����������xxxxxxxx��������������������zy{��������������zz�������)5;<94)��������������������������������������������������������������MORTabfa]TMMMMMMMMMM����������������������������������������US[\gtx{ztigf[UUUUUU�����5LRUSOB;)�������������NQY[cht����}xtlh[SON����)59EHF=5)��������

����		"/;B>;//"	������.0/(�������)55954))-6BEBB6)"#))))))	

#$'%##
				��������������������������������������������������������������������������������)*/6CIMIC6*�������#)26/#
����:789;=HUaiopmhaUH@<:RMNUbnnnicbURRRRRRRR�������
 ������������)<BB5)�����������
��������!����������������������������������������T[\_emz������zmaTOJT ���6BHPTO@6) xz�������������zxxxx
#(+++(#
P\g���������������gP������������������������

������������~||}����������������������

����zz���������������{z��~����������������)11,)&')*1-)�zÇÈÍÇ�|�z�y�r�o�z�z�z�z�z�z�z�z�z�z�Ľнݽ����ݽнĽ½½ĽĽĽĽĽĽĽĻ����������������������������������������������	���������������������������������� �����������������������������ʼּ׼ּּּؼּмʼ������ƼʼʼʼʼʼʿG�T�T�T�M�P�G�;�8�;�=�@�G�G�G�G�G�G�G�G���������������������������������������ݽ����ݽнĽýĽнؽݽݽݽݽݽݽݽݹܹ���������������ݹܹٹֹܹ���#�0�<�I�N�R�T�S�K�@�0��
����������Ŀ����������������������ĿĳĲĬĳĽĿĿE�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�Eټ��#�'�(�'�%��������	������Ŀ��������������ĿĸĿĿĿĿĿĿĿĿĿĿ�Z�f�s���������������s�f�X�Q�M�J�M�S�Z�ûƻлػڻлͻû������������ûûûûûÿy���������������y�o�m�l�m�r�y�y�y�y�y�yƚƧƳ�������������Ƴƚ�~�u�R�O�b�uƚ�.�;�G�T�`�m�y�{�y�j�`�T�G�;�.�&���"�.�M�Z�f�j�s�w�{�y�s�f�Z�M�A�:�4�0�5�A�J�M�����)�7�G�N�K�2�)������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��T�a�a�m�n�s�x�x�m�a�]�T�T�N�H�F�H�J�T�T���������������������������a�V�f�p�z���������ĿοϿ̿ĿĿ������������������������@�3�.�.�'�������'�/�3�@�@�@�@�@�@�����������ʾ����������������������������A�N�Z�a�g�l�g�`�Z�V�N�A�A�=�A�A�A�A�A�A�6�B�I�O�P�P�L�I�@�8�)�%����
��#�)�6����������������������������������������ÇÓàìùü��ûùìàÓÇ�~�z�z�zÄÇÇ�m�y�����������������y�m�g�b�e�i�m�m�m�m�N�[�g�j�a�T�:�)������������������)�N�m�����������������������z�m�e�`�_�a�f�m��������������������������	��"�/�7�?�D�D�?�;�/�"��	�����������	����#�/�8�4�/�"���
�����������������������������������������x�����������������N�Z�g�r�g�f�Z�N�L�N�N�N�N�N�N�N�N�N�N�N�Z�f�r�f�d�a�Z�X�M�K�M�R�Z�Z�Z�Z�Z�Z�Z�Z�-�-�:�F�I�N�F�:�-�)�+�-�-�-�-�-�-�-�-�-ŔŠŭŹ����������ŹŭŠŗŌņ�y�x�{ŇŔ�ûлܻ������������ܻлĻ��������
�����!�#�%�#��
������
�
�
�
�(�4�A�M�U�Z�f�n�r�f�Z�M�A�4�'�����(���������������������x�s�`�c�p�������������������������}�y�t�y�������������������G�J�G�E�>�;�.�"� �"�"�.�;�=�G�G�G�G�G�G��(�5�A�N�Y�W�N�K�A�5�(��������E�E�E�E�E�E�E�E�E�E�E�E�E�EuEuEmEnEuEuE����ɺֺںںֺκɺź������������������������'�@�P�X�T�G�4�'������������������������������������������������������g�t�x�t�n�g�f�c�g�g�g�g�g�g > . 6 , F R g ^ O / * * C f J  R # ; 3 : & - ` C ? W = < J > ' ; � 0 5 6 H T S L Z \ A ~ O v S ` 4 - C 8 S j    N  a  y  �  �  :  _  �  A    �  J  �  &  3  �  �  �    �  V  �  |  5  �  �  �  �  �    �  /  �  �  �  i  �  ?  d    ?  k  �  e  *  D  8  o  �  M  w  '  Q  +  �  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9  D9    �  �  �  �  �  �  �  {  c  c  z  �  �  �  ~  p  _  N  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �            +      �  �  �  �  �  �  �  �  �  �  �  �  �  �       >  \  �  �    j  �  �  �          �  �  �    :  �  g  �    ]  ?       �  �  �  �  g  Q  :  $    �  �  �  �  �    _  �  �  �  �  �  }  q  e  Y  M  E  B  >  ;  8  8  :  <  >  ?                  
        (  5  B  P  ]  j  w  �  �  �  �  �  �  �  ~  m  ^  O  @  2  #     �   �   �   �   �   �          
       �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  c  0  �  �  �  =  �    I   z  �    x  �  	$  	S  	v  	�  	�  	�  	]  	'  	  �  X  �      �  �  l  y  {  y  q  a  J  .    �  �  z  8  �  �  X     �  �  }  9  .  #    	  �  �  �  �  �  �  |  i  V  @  '    �  �  �  �  �    |  y  v  s  }  �  �  �  �  �  �  �      (  :  L  �  �  �  �  �  �  �  �  �  �  �  �  t  h  ]  Q  E  9  .  "  B  ^  f  e  c  r  �  �  �  �  �  �  �  N    �  k  (  �  �  B  E  H  H  G  G  G  E  C  @  =  7  .    �  �  �  �  �  �  K  J  J  E  ?  7  /  '             	    �  �  �  |      8  =  8  !    �          �  �  }  ,  �  H  �  �  M  9  2  .  ,  (        �  �  �  �  W  #  �  �  d    �  �  B  ]  j  r  u  v  t  o  c  N  3    �  �  �  O  �  `  �  m  �    �    9  N  b  h  a  X  |  �  A  �      
�  	R  �  �  �  �    ]  �  �  �     �  �  �  3  �  a  �  �  �  	�  h  �  �  �  �  ~  y  p  g  [  M  >  -    �  �  �  �  F    �  b  S  7    /  '      �  �  �  �  �  �  �  H  �  w  �  /  4  Z  Q  I  @  6  ,  !      �  �  �  �  ~  l  Y  H  <  /  "  �  e  I  2    �  �  �  �  I    �  �  �  �  �  �  v  \  B  1  >  E  A  :  1  #    �  �  �  �  t  N  '  �  �  \  &    �  �  �  �  �  �  �  y  d  O  8      �  �  �  �  �  i  N  	�  	�  
  
#  
;  
D  
>  
  	�  	x  	  �    �  J  �  �  �  �   �  e  �  �  �  �  �  �  �  x  d  H  $  �  �  J  �  {  <  �  -  W  g  e  M  4    �  �  �  �  �  d  G    �  �  �  j  S  9  �  �  �  �  �  s  a  O  :  $    �  �  �  �  t  H     �   �    �  �  �  x  T  D  2    �  �  r    �  Q    �  �  �   �  �  �  �  �  �  �  �  {  ]  ?    �  �  �  T     �  N  �  r  m  m  n  m  h  c  _  Z  U  P  J  D  =  6  -  "      �  �  	p  	�  	�  	�  	�  	�  	�  	�  	�  	�  	H  	  �  <  �    E  V  "  �    D  [  b  U  >  "  �  �  �  s  c  >  �  �  D  �  0  |  )  �  p  Y  D  )    �  �  �  [  (  �  �  �  e  4  �  �  �    ;  0  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  <  4  +  #      �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  s  r  �  �  �  �  �  �  �  �  �  �  �  �  �  Y    �  #  #    �  �  �  �  �  �  �  �  �  f  B    �  �  h  �  �    �  �  �  �  &  g  q  f  R  3  �  �  [  �  z    �       t  �  �  �  �  �  �  �    p  a  R  D  @  D  G  K  `  z  �  �  �  �  �  �  �  �  �  �  �  }  W  (  �  �  Z  �  �  �  r  �  �  �  �  �  �  �  s  b  Z  f  ^  ;    �  �  �  T  �  �  �    0  B  H  K  O  R  Q  O  L  H  C  ;  3  *  !      �  2  �  �  �  �  �  �  �  �  ~  o  `  Q  >  )    �  �         k  X  E  2    	  �  �  �  �  �  t  U  /    �  �  �  \  E  
5  
\  
Q  
C  
1  
  	�  	�  	n  	  �  I  �  S  �  '  �  �    �  �  �  �  	b  	�  	�  
  	�  	�  	�  	�  	>  �  �    i  �  �  �  �  �  	#  	�  	�  	�  	�  	�  	v  	N  	  �  �  #  �  /  �  �  �  �  �              	  �  �  �  �  �  �  �  �  �  �  w  i  [  �  �  �  z  [  ;    �  �  �  �  �  v  I    �  e    �  �