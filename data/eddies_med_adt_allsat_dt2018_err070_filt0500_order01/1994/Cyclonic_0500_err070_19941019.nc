CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�S����       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��a   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @FУ�
=q     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v�=p��
     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @R            �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��            6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �(��   max       ;ě�       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�|�   max       B5'�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�V�   max       B4�$       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >T��   max       C���       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >z�l   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��a   max       P��       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�oiDg8   max       ?�M:��       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       <���       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @FУ�
=q     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v������     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�I�^5?}     @  [L               D   0               	   �                  �   Y      _         .            }         
   &      	      �   2      
         	      
      	                        
         -               	      )          
   O"�N`��N� �Nc�kP���O�:�O@�N�	�N�@�P�SNv�LP��)N�!mNW�`O���O�VeN2�P�R{O�"3OT�Pv<N ThN���PIN���OLG'N���P���N�)�O6��O1%�P!�N���NI��O"��P �Pu��NܰN��N�>�O�iMNUW�Ni��Nj�O8�O$UuOV�N�FN�ŢO���N؅OU��Oa�>N�U�O9��O��fO�_�O�l�N�)O~;UM��aO�N���O��O�2�NʞeN�e�N=��<���<49X;�`B;D��;o��o��o���
���
�ě��ě���`B��`B�t��49X�e`B�e`B�u�u��o���㼛�㼣�
���
���
��1��1��9X��9X�ě����ͼ�����/��/��`B�����o�C��t��t���P��P���#�
�'',1�0 Ž49X�D���D���H�9�P�`�P�`�P�`�P�`�aG��ixս�7L��7L��O߽�\)��\)������������������������������

�����������V[agnt}�����{tg[ROVVbht������thcbbbbbbbb����������������{y|�z������������zrnmmpzaan���������zunda_^azz�������|zqzzzzzzzz)-6ABIIB6)������ �������������������������������naH<#
/<Unz���
#)$%##
������st�������tqrssssssss���������
��������h\OC6*" $*6:C\orlh#%/85/# z���������������zpszmw~��������������wsmcgt����������tgeb_^c��������	���������ABOWOOEB=9AAAAAAAAAA�������������������������������������������������������������
 ���������������������/H��������eW<
��X[hkltuzuth[UUUZXXXX�)6=<961' ��uz~�������������zvux��������������|wtux^amoonmmgbaYXY[\^^^^HHLUZYXUHHA@HHHHHHHH����������������������%/<?CH<<MJ</
����� 	#<n{��{nP<#������������������������%)66;96)%&%%%%%%%%%%#,/203/,##"/<EHPPJ@<8/#!"

"
	





#'0:<=<40#������������������������	��������TTamrvz}�zmaUTQPQTT��������������������MOSW[`hhhc[OMMMMMMMM��������������������/<HUgnrsnlrm[UH<7'*/#)+)$)*8BO[hsusnh[P=6)$�� )-75+������z|��������zxszzzzzzz��������������������{��������������uqsu{�����!'' ���������#0<UfkhbI<0
����������������������#)0<IRTTLI<0-'#.0020,$## ""������ }��������������|}}}}��������������������)B[t��xt[NB5��������������������)5<650/)<BFNZ[[\[ZQNIB<<<<<<���������������������
��*�/�:�>�/��
���ѿ˿ɿѿٿݿ�����ݿѿѿѿѿѿѿѿ���������������������
�����������M�I�C�A�M�T�Y�\�_�a�\�Y�M�M�M�M�M�M�M�M�(���ǽν��4�Z���������뾥�s�M�(�z�x�w�~ÇÖàìù����������ùìàÓÇ�z�N�M�E�D�I�Z�g�i�s�x�|�������s�n�g�Z�N���������������������������������������Żl�d�_�U�S�O�S�T�_�l�r�q�o�w�l�l�l�l�l�l�(��������(�5�N�a�g�q�u�z�x�g�A�(�s�r�s�w���������������������s�s�s�s�s�sE�E�E�E�F1FkFwFtFhFAFE�E�E�E�E�E�E�E�E��.�.�"����"�.�;�G�H�R�S�T�V�T�G�;�.�.�U�R�O�U�Y�a�g�n�x�y�n�a�U�U�U�U�U�U�U�UìàÝÖÑÇ�z�n�rÇÓàìù����ÿûùì�z�u�v�t�o�g�`�T�G�;�.�&��"�%�<�T�`�m�z�����%�)�6�<�9�6�)� ���������3�!��,�1�+�@�~��0�P�F�:�"��ֺ��~�L�3�û������������������ûܼ��(�-�+������������������������������ ���������˻���û��l�a�j����������@�c�f�Y�@�'����H�F�A�H�T�U�a�c�a�T�H�H�H�H�H�H�H�H�H�H�����������$�'�+�$�����������������������ؿݿ����(�A�N�g�����g�N�(���������������������������������������������������)�E�[�g�t�t�g�[�U�B�5�)���z¦®±¦���m�W�I�?�D�T�m��������������� �������T�O�K�T�`�k�m�y�����������y�m�`�T�T�T�T�����������������������ĿϿѿڿѿοĿ������ݿѿ����������Ŀпѿݿ�����������꿸�����y�������Ŀ�����#�(�1�(��ѿĿ����	���(�4�5�A�N�T�N�A�5�(������4�4�4�6�A�M�Z�a�Z�Z�M�A�4�4�4�4�4�4�4�4���������������ɾ׾�������ؾ׾ʾ���D�D�D�D�D�D�D�D�D�EE7ECESE[E[EVECEED��������s�[�O�@�E�N�s���������������������*�&�*�6�C�O�\�^�\�O�C�6�*�*�*�*�*�*�*�*ùõùù������������ùùùùùùùùùùàÝÔàäìù������������������ùìàà�
�����/�<�U�n�v�zÉÉ�n�a�H�<�/�#�
�@�@�4�(�4�@�M�Y�[�f�f�f�Y�M�@�@�@�@�@�@�������������#�%�������DVDMDIDHDIDVDbDgDfDbDVDVDVDVDVDVDVDVDVDV�������������������������������������������������������
��������������������(�)�5�B�N�Q�[�c�f�[�N�B�5�)���Ϲʹù������ùùùϹ۹ԹϹϹϹϹϹϹϹ��b�\�^�b�c�l�n�{łŀ�~�{�x�n�b�b�b�b�b�b���������������������׾��������ʾ����A�@�=�A�J�N�Y�X�R�N�A�A�A�A�A�A�A�A�A�A�g�^�g�s�|�������������������������s�g���������������������¾ʾ׾�׾Ͼ��������������������ĿǿοοſĿ���������������������ļĹĳĥĦĳĿ�������������������̻������û׻���'�?�Y�f�k�f�_�X�@�'���л���}�������ּ����!�-�/����ʼ�������9�!������(�-�:�F�L�T�Y�`�c�b�S�F�9�����������������ɺֺ����׺ֺɺ��������޽�������(�A�F�Q�K�A�4�����齫���������������ĽƽĽ��������������������{�������������Ľн��ܽнĽ���������ŭťŠŕŔŇŁŇŔŠŭŹż��žŹŭŭŭŭ�0�#��������
��#�0�?�L�U�b�{ŀ�v�b�U�0�Y�Z�Y�]�m�|ĆčĚĦĩģĠĠĚā�t�r�h�Y�t�n�m�p�tāčĚĞĚęĔčā�t�t�t�t�t�tÇÄ�ÃÇÓÔàìòììàÓÇÇÇÇÇÇ�z�s�n�e�a�`�a�n�x�zÇÈÇÄ�z�z�z�z�z�z f ( R ] k % 5 B c ' D 5 8 2 e J a i Q # b n . ^ T o B O P I > X � : D B J � > M Y i Z B C 8 J R ^ ) N � 8 S o | y X P M c p T S � 6 ? M  �  m    �  �  D  �  �  �  w  �    !  g  h  O  l  �    �  �    �  &  �  $  �    �  �  �  �    k  i    <  �  ?    ~  u  �  G  ;  `  D  ?  �  �  5  X  �  �        �    )    �  �  �  �  �  �  v;ě�;��
;o�D������H�9��1�D����t���j�u� Ĝ��1��o�'o��1�hs��������l��ě��ě���+��9X�'t���+������P�C���%�+�\)�0 ž(�ý��
�t��49X�]/��+�8Q�'@��L�ͽL�ͽ]/�D���u��hs�Y���O߽�+�y�#��%���w�ě����-�����\��\)���w���w��G���
=���;%�T�!��B��B�eB	�B[ZB �lB �7Be�B�.B��B?B`BW�B��B��B�BB0��B԰B��BAwB
~B��B�rB�8B�B�?B�hB�RB0�B��B��B ��B*2\A�|�Bz!B5'�B�B&Z�B��BnB��BW�B$�SB%r�Bg�B �A��}B��B.�Bl'Bp�B�B%�B�$B�B��B)��B-�B%�B ��B%̣B%��B`B
�7B��B	FB9�B�NB4�B��B�CB�B}�B 9TB ��BwB�>B�9B7�B?4B��B=rB��B��B1<KB�	BƢB@B
,zB�6BB�B��B>nB�BƢB��B�8B��BJ�B ��B)�YA�V�BvMB4�$B�SB&�\B��B@B��B>�B$��B%�PB��B�(A���B��B="B?2B��B��B@�BFwB��BB�B)��B-�NB%��B �6B&?�B%@]B=`B
�\B%�B@�BV�B��B7�A�b)A|ͯA�l�@��gA@��A��A�%A�k.@���A�H�A��C���Ab��AƪA���Af%�A�cm?��(@���A��o@��A�ldB��A��A�~1A�,A�YA��Ak4DAu|�A{(�A{^rA��rA<'�AQAkC�vA��0B �aAε/A��UA�_�@զ'@��TC�z�A�~BOA�$�>T��A��mAPL�A�Q?A��|AM�Aw|�A�K@�)A E�@} �@0�A40wA$pA$?�A�ȳA�q^AܛvAݳwA�ȒA��A��A}05A�}}@ڞ�AAȇÁA�!�Aπg@��A���A�tC���Ab��Aƃ�A�}Af��Aւ�?�=�@��A�2�@��A���B	9�A���A��BA���A��A�~VAk�Av��AyWRAz�YA�CfA;BPAOXC�y*A�ާB �1A��Ä́�A�uN@��@���C��A���B��A���>z�lA��AOvA�	A�{UAK 	Aw:A�@��A��@s�8@,��A7qA#A'��A�t�A�w�A�~�A݅�A�k�A�X�               D   1               
   �                  �   Z      _         /            ~         
   &      	      �   2               	      
      
                                 .               
   	   )   !      
                  K               '      @                  C   %      =         -            G            -            -   9                                       !                  -   /   !                  '   %                        5               '      7                  !         3         -            9            +               9                                                         -   /   !                  '            O(HN`��N� �Nc�kP?̽O�:�O!ͯN�	�N�@�P�SN@l�P�3<N��NW�`N��rO�VeN2�OѧGOvQhO5�P@QN ThN���PIN���O1]XN���P��N�)�O\�O�O��.N���NI��O"��O���Pu��NܰN��N�>�O�iMNUW�Ni��Nj�O8�O$UuOV�N�FN�ŢO�×N؅N��ON9�N�U�O9��O��fO�_�O�l�N�)O~;UM��aO�N���O��Oq��NʞeN�e�N=��  $  Z  �  g  g  	�  �  8  \  �  �    �  }  �  }  �  
{  
o    	�  �  �  C  �  �  �  	{  A  �  �    l    d  F  �    .  �  ;  g  _    0  �  E  I    ?  �  �  �  �  �  r  �  ~  �  �  �  �  �  �  {  �  �  <���<49X;�`B;D���e`B��o�ě����
���
�ě���`B���ͼ#�
�t����ͼe`B�e`B���P��w���
�+���㼣�
���
���
��j��1�L�ͼ�9X����������`B��/��/��`B���
���o�C��t��t���P��P���#�
�'',1�0 Ž8Q�D���Y��L�ͽP�`�P�`�P�`�P�`�aG��ixս�7L��7L��O߽�\)��\)���㽧�������������������������

�����������V[agnt}�����{tg[ROVVbht������thcbbbbbbbb��������������������z������������zrnmmpz`adnz��������zngba``zz�������|zqzzzzzzzz)-6ABIIB6)������ ��������������������������#/<Wp����maUH<#�� 	
 ###
�����st�������tqrssssssss��������������������h\OC6*" $*6:C\orlh#%/85/# �����������������������������������{yuu�fgt��������tlgeeffff����������������ABOWOOEB=9AAAAAAAAAA������������������������������������������������������������������������������������	#H�������n[I<#	X[hkltuzuth[UUUZXXXX #)56964.$� ������������z�������z�������������{xvvz^amoonmmgbaYXY[\^^^^HHLUZYXUHHA@HHHHHHHH����������������������
#)+,+(#
������� 	#<n{��{nP<#������������������������%)66;96)%&%%%%%%%%%%#,/203/,##"/<EHPPJ@<8/#!"

"
	





#'0:<=<40#������������������������	��������TTamrvz}�zmaUTQPQTT��������������������MOSW[`hhhc[OMMMMMMMM��������������������/<HUekmlnqlZUH<8((+/#)+)>BO[hmqpmhh[[TOJBA>>��)+2$������z|��������zxszzzzzzz��������������������{��������������uqsu{�����!'' ���������#0<UfkhbI<0
����������������������#)0<IRTTLI<0-'#.0020,$## ""������ }��������������|}}}}��������������������)M[dgt�{tgNB5��������������������)5<650/)<BFNZ[[\[ZQNIB<<<<<<�������������
��#�(�/�9�<�<�<�/�#��
���ѿ˿ɿѿٿݿ�����ݿѿѿѿѿѿѿѿ���������������������
�����������M�I�C�A�M�T�Y�\�_�a�\�Y�M�M�M�M�M�M�M�M�(������A������þǾξ˾������s�M�(�z�x�w�~ÇÖàìù����������ùìàÓÇ�z�Z�V�N�G�F�F�M�N�Z�g�s�y�}�~�{�s�g�d�Z�Z���������������������������������������Żl�d�_�U�S�O�S�T�_�l�r�q�o�w�l�l�l�l�l�l�(��������(�5�N�a�g�q�u�z�x�g�A�(���t�y����������������������������������E�E�E�E�E�E�E�E�E�F1FcFqFsFoFbF;FE�E�EͿG�C�;�.�"�!�"�#�.�:�;�;�G�N�P�O�G�G�G�G�U�R�O�U�Y�a�g�n�x�y�n�a�U�U�U�U�U�U�U�UìêàÜÚØÖßààêìøùÿüùïìì�z�u�v�t�o�g�`�T�G�;�.�&��"�%�<�T�`�m�z�����%�)�6�<�9�6�)� ���������L�@�5�6�D�L�Y�e�r�����������������~�e�L�û��������ûлܻ�����!������л����������������������� ���������������׻���û����v�������л����'�T�Q�.�'����H�F�A�H�T�U�a�c�a�T�H�H�H�H�H�H�H�H�H�H�����������$�'�+�$�����������������������ؿݿ����(�A�N�g�����g�N�(����������������������������������������������������)�5�B�[�g�[�R�N�B�5�)���z¦®±¦�����w�W�K�O�a�m�������������������������T�O�K�T�`�k�m�y�����������y�m�`�T�T�T�T���������������������������Ŀ̿ѿǿĿ��������������Ŀѿݿ������ݿѿĿ��������������|�������Ŀ������(����ѿĿ����	���(�4�5�A�N�T�N�A�5�(������4�4�4�6�A�M�Z�a�Z�Z�M�A�4�4�4�4�4�4�4�4���������������ɾ׾�������ؾ׾ʾ���D�D�D�D�D�D�EEE*E7ECELEOEKE=E7E*EED��������s�[�O�@�E�N�s���������������������*�&�*�6�C�O�\�^�\�O�C�6�*�*�*�*�*�*�*�*ùõùù������������ùùùùùùùùùùàÝÔàäìù������������������ùìàà�
�����/�<�U�n�v�zÉÉ�n�a�H�<�/�#�
�@�@�4�(�4�@�M�Y�[�f�f�f�Y�M�@�@�@�@�@�@�������������#�%�������DVDMDIDHDIDVDbDgDfDbDVDVDVDVDVDVDVDVDVDV�������������������������������������������������������
��������������������(�)�5�B�N�Q�[�c�f�[�N�B�5�)���Ϲʹù������ùùùϹ۹ԹϹϹϹϹϹϹϹ��b�\�^�b�c�l�n�{łŀ�~�{�x�n�b�b�b�b�b�b�����������������;׾��� �����ʾ������A�@�=�A�J�N�Y�X�R�N�A�A�A�A�A�A�A�A�A�A���������������������������������������������������������������ʾ׾߾׾;��������������������ĿǿοοſĿ���������������������ļĹĳĥĦĳĿ�������������������̻������û׻���'�?�Y�f�k�f�_�X�@�'���л���}�������ּ����!�-�/����ʼ�������9�!������(�-�:�F�L�T�Y�`�c�b�S�F�9�����������������ɺֺ����׺ֺɺ��������޽�������(�A�F�Q�K�A�4�����齫���������������ĽƽĽ��������������������{�������������Ľн��ܽнĽ���������ŭťŠŕŔŇŁŇŔŠŭŹż��žŹŭŭŭŭ�0�#��������
��#�0�?�L�U�b�{ŀ�v�b�U�0�[�[�_�t�}āăĉčĚģģĞĞĘā�{�t�h�[�t�n�m�p�tāčĚĞĚęĔčā�t�t�t�t�t�tÇÄ�ÃÇÓÔàìòììàÓÇÇÇÇÇÇ�z�s�n�e�a�`�a�n�x�zÇÈÇÄ�z�z�z�z�z�z U ( R ] c % , B c ' = 3 , 2 f J a ( E  c n . ^ T ` B Y P J ? P � : D 0 J � > M Y i Z B C 8 J R ^ , N < 4 S o | y X P M c p T S � 6 ? M  g  m    �  �  D  k  �  �  w  l  d  �  g  1  O  l  �    *  �    �  &  �  �  �    �  Q  I  q    k  i    <  �  ?    ~  u  �  G  ;  `  D  ?  �  s  5    �  �        �    )    �  �  �  �  �  �  v  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  #      �  �  �  �  �  s  I  #  �  �  %  �  �  H  �   �  Z  U  Q  L  F  ?  9  0  '      �  �  �  �  �  X     �   �  �  �  �  �  �  �  �  �  �  ~  |  z  s  j  a  X  V  W  X  Y  g  X  J  ;  +      �  �  �  �  �  �  �    o  `  O  ?  .  �  #  D  K  _  b  P  ;  '    �  �  �  �  u  [    �  �  �  	�  	n  	=  	  �  �  ^    �  �  D  �  �    �    g  k     i  �  �  �  �  �  �  �  �  �  �    h  J  '  �  �  �  m  ;    8  &       �  �  �  �  �  �  h  N  6    �  �  �  U  (   �  \  R  E  6  '  (  .      �  �  ~  C    �  o  #  �  ~  )  �  �  �  �  �  o  X  C  $    :  @  8  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  S  6    �  �  m  ,   �   �   ^  ;  �    �  �  M  �  �  f  �  E  �  
�  
1  	v  �  �  �  �  F  c  t  �  �  �  �  �  �  �  �  �  u  c  L  1    �  $  �    }  w  r  j  _  S  A  +    �  �  �  �  l  :  �  �  �  C    �  �  �  �  �  �  �  �  �  �  �  n  G  !    �  G  a  �  �  }  w  l  c  Y  R  H  8       �  �  �  i  @    �  z  .   �  �  �  �  �  �  �  ~  b  G  %    �  �  6  �  �  ~  C     �  �  	  	E  	]  	Y  	]  	�  
  
o  
{  
a  
0  	�  	~  �  O  m  5  p    �  	g  	�  
  
T  
m  
g  
J  
/  
  
  	�  	�  	�  	�  �  �  �    �  �                  	  �  �  �  �  �  b  .  �  �    	>  	m  	�  	�  	  	s  	e  	?  �  w  �  �  E  �  �  �    6  4  �  �  �  �  �  |  o  b  T  N  Q  U  Y  O  =  *      �  �  �  �  �  �  �  �  �  �  x  n  c  Y  P  G  >  4  ?  P  b  s  �  C  =  '    �  �  �  �  y  I     �  �  2  �  C  �  �  �   r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  r  m  g  �  �  �  �  �  o  M  $  �  �  �  w  K     �  �  �  �  �  �  �  �  �  w  \  =    �  �  �  }  Q  %  �  �  �  �  �    �  �  	
  	D  	j  	z  	r  	T  	#  �  �  `    �  |  ;    ^  [     �  A  <  6  1  +  $       �   �   �   �   �   �   �   �   �      o   _  �  �  �  �  �  �  �  �  �  �  �  �  �  h  N  3        D  �  �  �  �  �  �  x  h  X  F  2      �  �  �  �  �  p  U  
      �  �  �  �  �  x  V  2    �  �  k     �  O    e  l  \  L  <  &    �  �  �  �  �  e  D     �  �  �  W     �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  V  E  4  $      �  �  �  �  �  �  d  ?    �  �  �  @  �  /  U  (  �    =  D  )     �  h  �  �    �  N  �    ,  �  �  �  �  d  ?    �  �  �  a  -  �  �  e    �  7  �  >         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  .  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  g  �  �  �  z  e  H  %    �  �  �  �  �  c  9    �  �  �  O  ;  4  !    �  �  �  w  o  �  �    �  �  �  �  �  d    �  g  ]  R  N  N  M  G  @  1      �  �  �  �  �  z  \  ;    _  Z  U  P  J  D  9  .  #         �  �  �  �  �  �  �  �       �  �  �  �  �  �  t  ]  A  +                  0    	  �  �  �  �  �  |  _  @    �  �  �  �  �  O    �  �  �  �  �  �  {  a  C  $    �  �  �  �  �  p  :     �   �  E  1      �  �  �  �  �  i  W  5    �  �  s  7  �  �  z  I  D  @  ;  7  2  -  "      �  �  �  �  �  �  r  X  >  %        �  �  �  �    =  �  �  ]  	  �  a    �  �  %   o  +  ?  >  >  =  ;  9  6  0  -  !    �  �  �  j  "  �  �  )  �  �  }  y  u  o  i  c  \  R  H  >  4  )        �  �  �  �  �  �  w  �  �  �  q  C    �  �  z  E      �  �  �  �  �  �  �  �  �    H    �  �  |  Y  9    �  �  Q  �  �  0  �  �  �  w  c  K  4      �  �  �  �  �  y  ]  ?     �  �  �  �  y  e  m  m  J  '    �  �  �  �  �  Y  .    �  �  p  r  e  ^  V  J  4    �    .  !  �  �  �  X    �  �  8  �  �  �  f  N  Z  �  �  �  u  I    �  �  h    �    h  �  �  ~  \  9    �  �  �  �  z  m  V  5      �  �  �  �  +  A  �  �  �  �  �  �  �  k  G    �  �  �  �  �  v  G    �  l  �  �  �  r  D    �  �  �    _  7    �  i  �    �  :   �  �  �  �  �  �  �  v  l  d  ]  U  N  F  B  F  J  N  R  V  Z  �  �  �  �  �  �  �  �  �  �  �  u  _  E  &    �  �  �  ~  �  �  �  �  |  f  P  ;  $    �  �  �  �  �  o  O  9  1  *  �  �  �  �  �  �  X  +  �  �  {  ,  �    z  �  �  k  �  \  7  i  w  o  j  X  5     �  �  �  H  �  �    �  '  �  U  �  �  �  �  �  �  �  �  �  �  ~  \  5    �  �  �  a  .  �    �  �  �    n  [  G  1      �  �  �  �  �  �  �  �  e  I        
      �  �  �  �  �  �  �  �    k  [  J  :  *