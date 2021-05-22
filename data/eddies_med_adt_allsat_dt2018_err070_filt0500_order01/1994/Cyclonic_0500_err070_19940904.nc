CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�ȴ9Xb       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N_   max       P�+B       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��G�   max       ;�`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F������     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vw
=p��     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @��`           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��u   max       �#�
       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�w'   max       B4��       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�y   max       C��h       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Ns   max       C��^       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N_   max       P��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�R�=   max       ?Ӄ{J#9�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��G�   max       ;��
       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F������     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vw
=p��     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @���           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�;dZ   max       ?�l�C��     �  ]                  -            
                  
   O   	      &               &               -   
               #                  5   '                     6         9      7      Y            .   	      )   $         #   
   Oq�N���N��N�h{N�<@O�o,O���Nqw)N�ˀN=�pN�%eN�v�N�ZNY�N�pN�6P�+BNa< O1j�O���O-�O]ND�N���O���NK�Oʲ�O��No@�PdYN��N��iN.*yN�I�O���P�{�N.M2N�z�O	TjN_N�FTP%L�P&��OTO��N1��N�,ONn�IO�<�P
c�O)O$��O��O�PE-O�ZP.MN�rO�g�N͔P`4NI��N��9P+�~O��N�EXN���O'�NRf�N�RS;�`B��o���
���
���
�ě���`B�t��t��49X�D���T���T���T���e`B�u��o��C���t���t����㼛�㼣�
��1��9X��9X����������/��h��h���o�o�o�o�+�+�+�C��t��t����#�
�#�
�#�
�'''T���Y��Y��]/�]/�e`B�y�#�y�#��+��7L��\)��\)��t����-���罬1�� Ž�j��j���ͽ�G������%.-/,��������������������������	��������U[_gqt�����tg[UUUUUU'(*-6COXTPHC<6*#�zploz����������������������������������\amz�}zmaY\\\\\\\\\\/6BHOOSXOB<60+//////�����������������������������������������������������������������������������������%!�����������������������������������������������y����������������yuy!#/<?HIH<3/-#"!!!!!!�������������������������)3������������ ����������������������������#.03<>=<50.#egitz�����trgdeceeee��������������������otw������toooooooooo)=O[cmpnhZOB��������������������	!"	����*'���������168BFOZV[][OB>651111�������������_amz}zxma\__________LOU[ghqphh^[ZOKILLLLntv������������|tpln��%0HU{������{U<������������&)6;:62)%EOW[htz�������th[UOE:;HKLHF;6:::::::::::egirt���������trkjge���#/<<861#����������������������blt�����������tgb`b��������������������
####

����������������������������������������nq{����������{vomonn��������������������������������������������� 
�������#0<IV^_[MI;#
/04:<@IJRUVXWRI<00//0<HU���������zaU</'0��������������������Sanz����������n^RNNS��������������������������������������������������!(-.,)�������������������������Vansz��ztna`YVVVVVVVB[gt���tbWN5)�����
#)##
����������������������������
��������)5;BEKNB5)!���
������������,/38<<HKPUPH<7//,,,,�0�/�)�$���
���$�0�=�I�Q�R�N�E�=�2�0�H�F�?�C�H�I�U�a�c�a�_�]�Z�U�H�H�H�H�H�H���������������������¾������������������������������������������������������	������׾;¾��ʾ׾��������	��	�=�1�$����������������$�0�7�<�>�?�=�T�N�H�/�� �#�5�;�H�K�T�a�m�s�t�}�g�d�T������������� ��	����������������������S�K�M�S�_�_�l�x���x�r�l�_�S�S�S�S�S�S�Y�U�M�E�M�U�Y�]�f�n�r�v�r�f�Y�Y�Y�Y�Y�Y�;�:�.�(�'�.�;�G�P�T�Y�`�a�`�T�G�;�;�;�;�����(�)�4�5�4�(�������������|�s�j�h�s�����������������������������������������������������������������ʾ̾׾߾���׾ʾ������!� ���!�-�:�F�L�S�T�S�K�F�:�-�!�!�!�!���~�Q�?�A�~����-�F�]�o�l�F�-�����ɺ��H�E�?�>�H�S�U�V�\�a�c�a�W�U�H�H�H�H�H�H�*�'�)�*�6�B�C�O�\�^�h�r�s�o�h�\�O�C�6�*�����չƹ��չ������'�3�B�@�;�3�'��"��	���	���"�/�:�;�B�H�J�H�=�;�/�"�q�g�Z�N�P�_�g�s�����������������������q�M�E�@�9�4�2�4�;�@�M�M�M�L�M�Q�U�M�M�M�M�ѿοѿ׿ݿ�����	��������ݿѿѿѿ�������ĿĳĘčĚĠĦĿ��������#��
��������������	�������������L�F�H�g�u�z���������������������s�g�Z�L����ѿ������������������Ŀݿ����������������������������������������������	�� ��.�;�m�y���������{�m�`�G�.�"��T�L�H�<�;�5�;�H�R�T�U�[�`�a�b�a�T�T�T�T���������~�~��������������������������������������� �������������ù����������ùϹܹ޹����ܹϹùùù��/�'�"���
�����������/�G�W�T�J�H�;�/���������s�c�N�E�6�7�Z���������������������������������������������������������������������ĿͿѿڿѿĿ��������������������~�{�y�x�y�{��������������������������������������������������������������������ƾ���������������������������������������������)�B�Y�_�b�m�[�U�B�)���������!�:�F�S�_�l�n�l�_�:�!���㺽���*��������#�)�*�+�1�C�E�J�H�C�6�*�6�,�*�&�*�3�6�C�O�\�h�n�u�u�h�b�\�O�C�6ŹŶŹŽ��������������������ŹŹŹŹŹŹ�b�X�[�b�i�n�{Ņŀ�{�v�n�b�b�b�b�b�b�b�b�����������������ûŻû������������������_�S�C�>�D�P�_�l���������߻ܻл����l�_�����#�0�I�V�j�s�t�q�]�V�I�=�$�����B�8�6�3�1�3�6�?�B�E�O�V�[�d�e�\�[�O�B�B���׾־;ʾǾʾʾξ׾��������𽞽����~�y�y�~���������ݽ���۽ѽĽ����4�(�����������(�4�A�M�V�M�L�A�4�����������������ʾ׾������޾׾ʾ��n�m�e�c�h�n�x�{ŇŔśŞŖŔōŇ�{�p�n�n�z�f�]�X�X�^�m�������������������������z����ݼ�����������������������������ķĳĨ��������#�2�>�G�G�@�0�#���!����������!�.�/�5�9�.�&�"�!�!�_�G�<�G�N�l�������ݽ������ѽ����y�_àÛÓÇÂÇÓàìðìçàààààààà�����������Ŀѿ׿ݿ�޿ݿѿĿ������������j�T�O�@�4�1�<�H�U�Z�nÇÓèù����ùÓ�jD�D�D�D�D�D�D�EEEEE%E*E1E*E%EEED�����¿³°²·¿���������������������������������%�)�,�6�=�6�4�)������B�@�B�G�L�U�[�c�h�tāċć��t�h�[�O�H�B�'���"�'�4�@�D�@�=�4�(�'�'�'�'�'�'�'�'FFE�E�E�E�E�E�E�E�FFFFFFFFFF @ n < [ q + : N E ; Q U - i ; % g B 5 V - Y j X K X f ] R E R > C * � k e W V _ ? F l Y M v R Q _ , < �  F : = 9 r b F g X ; ~ 5 < T p T B    �  �  �  P  F  @  �  �  `  �  �  �  �    �  �  �  �  �  A  m  �  �  g  \  A  �  |  �  �  �  G  �  1    ^  �  c  $      �  �  ;  �  �  �  �  f  S    �  Q  �  :  �  C  >    l  ]  �  
  K    �  �  |  ȼ49X�T����o�#�
�T���P�`���
��o��1��1�u���㼓t���o��/�ě�������ͼ����e`B��/�o�ě������u���ͽ]/�0 ż����P��w��P�\)�D���@���7L�\)����w�t��<j��9X���-�<j�L�ͽ<j�@��T����\)��
=���
��\)��G����-��S���t���u�Ƨ�\���T��h���T��vɽ��m��F������J��;d�B�FBd.B*^B	�xB0��B
�B۶A�z�B)�B ��B"InBjBq�B�0B4�B,i�B#�BQB�rB�-Bt�B�B%uSB	ȠBE,B��B�B*avA���B��BP�B]A�k�BMLB ETB&aWB��B,�B�SA���B
vB�B"�;B
%B6AB"B��B�#B)o�B��B{�BlB%�!B&}IBr0Bx?B�_B��BgNBܥB�BH�B�LB�B�{B!�B6xB�yBW�B�B��B��B?]B	�HB0�?B>B��A�Q!B�BB �B!�B� BR�B�?B4��B,z�B>�BA�B��B�WB��B>�B%�B
-�B��B�B��B*��A�w'B�BB?�B]�A�{�BA(B ��B((EB�bB>�B:ZA���B	�B?�B"��B	�@B?�B?�B�	B �B)u�B[�B@!BƤB%�B&D�B=�B��B�GBF(B��B�VB�BH5B��B��B�2B?0BA�B@	B��B.B
;�A���AIc�A�$�AT�B	_�A��A���@���@�*AdA4^�A�`�A�upAN�@xP�@=d�A�-B'I?dn�A��_A�-�@�/�A�A�Y8A0��A��A{!�A�d�Ae�A�
�A��\A�
�>�yA�-A�_
A�
�Ax�Apl�A�q:B�A��p@`�?B LB%�A�^A�z�@�:%@�i�B
��A��QAT� A#y�A7��AO&�A���A��RA<A��AA#kA���AyJA��|C�_dA���A�x$Aۊy@�_,C��hB
<6A�ߞAJƁA�F�AS B	>�A�k�A曽@��t@��Af��A3 A�|}A�|AN�@w\)@1u:A�^xBrT?rR�A���A�{u@� �A��A��A0�JA��%Ay��A��Ab�A� �A���A`'>NsA�~�A�;A�W�Aw]Ap�A�eJB 0A�R�@k��B DyB�@A�~�A�fV@��-@�z�BM�A�e�AS!�A"�;A8�6AO�A�:A��AcA�uA�6A�GA�?�Az��A�G�C�X;A���A�{�A�x�@��
C��^               	   .                              
   O   	   	   '   	            '               .                  $                  5   (                     6         9      8      Y             /   
      )   $         $   
                        !                              G         '               +      %   #      %               #   =                  -   3                  +   %         #      '      '      #      -         3                                                                     A                        %      %   #      !               #   =                  #   3                  %   %               %      !      !      +         3                  OJ�zN{�"N���N�h{N.~�Ow��N�t�Nqw)N�ˀN�N�%eN�v�N�ZNY�N;�&N���P��N(�4O1j�OcIN���O]ND�N���O�6�NK�Oʲ�O��No@�O���N8$N��iN.*yNI��O���P�{�N.M2N�z�O	TjN_N�FTO�aP F�N���O��N1��N�,ONn�IO܄AP
c�N럝N�2*O�޼N�o�P=�N�O؈�N�rO�wN��P��NI��N��9P+�~O��N�EXN���O3NRf�N�RS  K  c    
  �    e  �  �    �    #  �  >  �  ^  �  �  �  >  +    3  �  �  �  �  �  r  F  ~  
  s  �  �  �  �  *  �    9  �  �  .  T  L  K  �  /  �  h  o      P  
�  	�    5  4    �  �  
  o  �  �  N  �;��
�D����`B���
�o�D���e`B�t��t��D���D���T���T���T�����㼋C���1��t���t���`B���
���㼣�
��1��j��9X����������/�t������o��P�o�o�+�+�+�C��t��49X��w�'#�
�#�
�''0 ŽT���e`B�ixս}�e`B�ixս�%���T��+��C���t���t���t����-���罬1�� Ž�j��vɽ��ͽ�G�����")(,)'�������������������������������������U[_gqt�����tg[UUUUUU36BCEMJC@61033333333xz��������������urx��������������������\amz�}zmaY\\\\\\\\\\/6BHOOSXOB<60+//////�����������������������������������������������������������������������������������%!�����������������������������������������������|����������������|w|"#/<<E<0//##""""""""������������������������	���������������������������������������������#.03<>=<50.#egitz�����trgdeceeee��������������������otw������toooooooooo)=O[cmpnhZOB��������������������	!"	���$ ��������36:BNOOQXOB633333333�������������_amz}zxma\__________NO[[\hmkh^[WOMNNNNNNntv������������|tpln��%0HU{������{U<������������&)6;:62)%EOW[htz�������th[UOE:;HKLHF;6:::::::::::egirt���������trkjge�����#/453/#
��������������������egpt����������tgdbee��������������������
####

����������������������������������������s{�����������{ropops���������������������������������������������

������
#0<OY[VI<0#

/015;<IPUWVUQIF<30//,0<HUaz������zaU<0(,��������������������[cnz����������n`XVW[������������������������������������������������� *,,+%���������������������������Vansz��ztna`YVVVVVVVB[gt���tbWN5)�����
#)##
����������������������������
��������),5:BEKMB5,)!���
������������,/38<<HKPUPH<7//,,,,�0�.�$������$�0�=�E�I�O�P�L�I�C�=�0�U�I�H�A�E�H�K�U�a�b�a�]�\�X�U�U�U�U�U�U�����������������������������������������������������������������������������׾Ծʾʾʾ׾�������׾׾׾׾׾׾׾���
�������������$�0�2�9�;�;�8�0�$���T�M�H�;�9�7�;�H�T�_�a�g�b�a�T�T�T�T�T�T������������� ��	����������������������S�K�M�S�_�_�l�x���x�r�l�_�S�S�S�S�S�S�Y�V�M�J�M�X�Y�Z�f�l�r�r�r�f�Y�Y�Y�Y�Y�Y�;�:�.�(�'�.�;�G�P�T�Y�`�a�`�T�G�;�;�;�;�����(�)�4�5�4�(�������������|�s�j�h�s�������������������������������������������������������������ľʾҾʾľ������������������-�-�!���!�-�:�F�I�O�F�:�1�-�-�-�-�-�-�����~�W�I�Y�~���ֺ�-�V�h�c�:�-���ɺ��H�G�@�G�H�U�Z�a�b�a�U�U�H�H�H�H�H�H�H�H�*�'�)�*�6�B�C�O�\�^�h�r�s�o�h�\�O�C�6�*�����ܹӹܹ�����'�)�3�5�7�8�3�'�����"���	���	���"�/�3�;�=�G�;�8�/�"�"�q�g�Z�N�P�_�g�s�����������������������q�M�E�@�9�4�2�4�;�@�M�M�M�L�M�Q�U�M�M�M�M�ѿοѿ׿ݿ�����	��������ݿѿѿѿ���ĿĳęďĚĦĿ�������
��!��
�������̾����������	�������������L�F�H�g�u�z���������������������s�g�Z�L����ѿ������������������Ŀݿ�������������������������������������������������.�G�T�m�y�}�������y�m�G�;�.�"��T�O�H�>�;�;�;�H�T�Y�^�^�T�T�T�T�T�T�T�T���������~�~��������������������������������������� �������������ù¹��������ùϹչܹ޹ܹ׹Ϲùùùùù��/�'�"���
�����������/�G�W�T�J�H�;�/���������s�c�N�E�6�7�Z���������������������������������������������������������������������ĿͿѿڿѿĿ��������������������~�{�y�x�y�{��������������������������������������������������������������������ƾ���������������������������������������������)�A�U�Y�Z�Q�H�B�)���������!�:�F�S�_�h�m�l�_�:�!���ຽ���*�'������'�*�,�6�?�C�D�I�F�C�6�*�*�6�,�*�&�*�3�6�C�O�\�h�n�u�u�h�b�\�O�C�6ŹŶŹŽ��������������������ŹŹŹŹŹŹ�b�X�[�b�i�n�{Ņŀ�{�v�n�b�b�b�b�b�b�b�b�����������������ûŻû������������������S�G�@�F�Q�_�l�����ٻ���ڻл����l�_�S�����#�0�I�V�j�s�t�q�]�V�I�=�$�����B�:�6�4�2�5�6�B�O�R�[�b�c�[�Y�O�B�B�B�B���׾оʾɾʾξξԾ׾ݾ�������㽞�������}�}���������׽�ݽڽԽȽĽ������4�0�(���
����(�4�7�A�K�M�T�M�F�A�4���������������������ʾ׾�����ݾ׾ʾ��n�n�h�k�n�z�{ŇŔŖśŔŒňŇ�{�n�n�n�n�z�m�f�`�a�f�m�z�����������������������z����ݼ�����������������������������ĺĴ��������#�0�<�F�E�<�0�#������
�����!�!�"�.�1�2�.�#�!�����`�G�R�l�������ݽ���	�
����ݽ̽����y�`àÛÓÇÂÇÓàìðìçàààààààà�����������Ŀѿ׿ݿ�޿ݿѿĿ������������j�T�O�@�4�1�<�H�U�Z�nÇÓèù����ùÓ�jD�D�D�D�D�D�D�EEEEE%E*E1E*E%EEED�����¿³°²·¿���������������������������������%�)�,�6�=�6�4�)������O�H�M�O�U�[�c�h�tāĊćā�~�t�k�h�[�O�O�'���"�'�4�@�D�@�=�4�(�'�'�'�'�'�'�'�'FFE�E�E�E�E�E�E�E�FFFFFFFFFF * x # [ N ' > N E D Q U - i J " c D 5 O 3 Y j X G X f ] R G T > C # � k e W V _ ? D l > M v R Q ] , / m  = ; 6 < r ^ > d X ; ~ 5 < T ` T B  �  �  �  �  ?  �  �  �  �  M  �  �  �  �  o  �  ?  T  �  �  �  m  �  �  Q  \  A  �  |    s  �  G  V  1    ^  �  c  $    s  u  !  ;  �  �  �  v  f    @  x    k  �    C  �  �    ]  �  
  K    �  ;  |  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  )  E  K  I  ?  1  !      �  �  �  �  Q    �  �  W  �   �  R  W  ]  b  a  ^  X  N  B  (    �  �  �  �  T    �  e  �  �  �  �     �  �  �  �  �  �  �  q  Q    �  �  _    �  �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  }  _  <     �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  s  a  M  5                �  �  �  �  q  )  �  m    �  �  +  �  �        '  /  <  K  T  Z  _  d  d  `  S  -  	  �  �  �  J  �  x  g  W  F  6  %      �  �  �  �  �  �  �  �  l    �  �  �  �  q  _  L  9  %      �  �  �  g  -  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  k  S  9       �  �  �  �  �  �  �  �  |  s  i  ^  R  G  ;  0  1  K  e  ~  �  �  �        �  �  �        0  K  r  �  �  �    I  z  �  �  #            �  �  �  �  �  �  �  �  �  �  i  M  2    �  �  �  �  �  �  �  �  �  u  X  ;      �  �  �  �  �  v  �       $  .  5  :  =  =  <  9  1  %    �  �  Y  �  �  6  �  �  �  �  �  �  �  �  �  �  v  j  ]  P  B  5  (        M  ^  Y  H  8  $  "  �  �  �  �  N    �  (  �  �  3  V    g  q  |  }  x  s  k  b  W  K  ?  2  %      �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  v  i  T  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  �  D    �  �  �  �  4  �  �  �  "  .  9  =  =  ;  3  *      	           �  �  �  �  �  +  &      	  �  �  �  �  �  �  �  �  ~  R    �  F   �   }       �  �  �  �  �  �  �  �  �  �  v  T  2      �  �  �  3  .  )  $        �  �  �  �  �  �  ^  7     �   �   �   s  �  �  �  �  �  �  ~  `  ?  %  
  �  �  �  X    �  i    w  �  �  �  �  �  �  �  �  �  �  �  �  ~  z  v  s  o  k  g  d  �  q  \  B  .  "    �  �  �    I    �  �  Y    �  {  F  �  �  �  {  ^  @  !  �  �  �  �  {  W  2    �  �  �  P    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  g  \  R  G  A  Y  j  r  p  e  M  -    �  �  J  �  ~    ~  �  �  �  b  >  >  >  C  K  _  l  e  _  [  V  Q  H  <  1  &        �  ~  |  {  z  w  s  p  k  g  c  ]  W  Q  G  8  (         �  
     �  �  �  �  �  �  �  �  �  �  �  �  �    t  i  ^  S  0  F  W  e  n  r  r  i  Z  I  3    �  �  �  �  ~  Y  /     �  {  j  W  A  *    �  �  �  �  i  9  #  A    �  �    �  �  �  X  '  �  �  �  J    �  �  L  �  �  G    �  �  L   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  f  O  8     �   �   �  *  !        �  �  �  �  �  �  �  }  j  W  C  /     �   �  �  �  �  �  �  �  �  �  {  t  j  \  N  @  2  %    	   �   �    �  �  �  �  �  �  �  �  x  i  ]  P  B  :  3  2  2  f  �  �     6  9  5  '    �  �  �  `    �  P  �  g  �  /  :   �  �  �  �  �  }  N  '    �  �  �  �  y  9  �  l    �  v  �    �  �  �  �  �  ~  s  g  \  N  ?  0       �  �  �  �  �  .  '                �  �  �  {  P  $  �  �  �  �  �  l  T  e  u  �  _  .  �  �  �  _  (  �  �  �  V  '  �  �  �  l  L  D  =  5  -  %          �  �  �  �  �  �  y  f  S  @  K  D  >  :  4  +  #        �  �  �  �  �  �  �  �  W    �  �  �  �  �  �  �  �  m  L  $  �  �  Z       �  �  G    /      �  �  �  ^  ,        �  �  e    �  !  �  �  0  U  g  �  �  v  g  R  /  �  �  |  +  �  {  .  �  �  =  �  �  9    2  :  U  ^  C    �  �  �  �  R    �  �  �  r  3  �  L  f  m  n  d  R  ;    �  �  �  P    �  [  �  }  �  )  �    
  	  �  �  �  �  S  #  �  �  �  Q    �  �  L  $  �  �        �  �  �  �  �  b  3  �  �  �  �  e    �    =  L  '  5  B  L  P  O  H  ?  0      �  �  �  �  Z  2  	  �  �  
=  
T  
i  
w  
�  
�  
o  
P  
*  	�  	�  	a  �  l  �  B  �  p    K  	�  	�  	�  	s  	E  	   	H  	_  	#  �  �  -  �  o    �  Q  Z  �  M          �  �  �  �  �  v  A    �  s  '  �  �  Z  (  �  /  0  1  4  4  1  $       �  �  �  �  �  ^  7    �  �  �  /  0    �  �  �  �  �  �  �  �  n  L    �  �  0  �  y  �    �  �  �  �  v  P  &  �  �  �  X  !  �  �  ?  �  �  �  u  �  �  �  c  B    �  �  �  p  D    �  �  �  b  !  �  �    �  n     �  �  �  �  �  �  �  �  I    �  h    �  "  �    
  	�  	�  	�  	w  	L  	  �  �  �  b  +  �  �  <  �  R  �  G  �  o  o  i  Z  F  /    �  �  �  �  �  \  4    �  �  X    �  �  �  �  �  g  F  #  �  �  �  �  ^  -  �  �  1  �  a  �  |    �  �  �  r  S  %  
�  
�  
;  	�  	^  �  j  �  r  �    C  �  N  F  ?  )    �  �  �  �  �  h  D    �  �  m  *  �  �  X  �  �  �  �  �  �  �  �  u  `  L  8  *    
  �  �  �  �  �