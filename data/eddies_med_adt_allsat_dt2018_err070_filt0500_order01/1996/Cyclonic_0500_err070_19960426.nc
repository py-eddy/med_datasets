CDF       
      obs    O   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���S���     <  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�V�   max       Pp!�     <  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��7L   max       <#�
     <   $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�=p��
     X  !`   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v���
=p     X  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  :   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          <  :�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       ;��
     <  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��2   max       B4��     <  =(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B4��     <  >d   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C���     <  ?�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�ہ   max       C�э     <  @�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          l     <  B   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     <  CT   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     <  D�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�V�   max       Pd�'     <  E�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U2a|�   max       ?�&�x���     <  G   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���w   max       <#�
     <  HD   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�=p��
     X  I�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v���
=p     X  U�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  b0   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         <  b�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B   max         B     <  d   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�������   max       ?�'�/�     �  eH   %         
      	   l               ,            	      '            
   !                  
   B      /      
                        	   9            :                        
         "                   K                  
      	               '      O���N!�O�N�eO��O��PN�
O
��N�syN&�&NC�O�|�N_�O���O�O�NvK�P&|DO��O�r�O�j�N�EwN��4On{O{�N�=~P+�O`��ND�BOd��Pp!�N��O�p�Nn��NJVZN]��N	CN]<�N�Y?N�(O��OI�>O62�Pr�M��NbK�O�2P.fN-�|M�V�O�IO��Oj0"O�r�NcXN�C�O�hN�ûOϐKO�*'OL��OSȹN�~xN��O��N�j�O���NߓLN^��Oxq�N��N�(�O
u�O DhOskOzׂN�W�NٿFN�9�Nrql<#�
<o;�`B;ě�;�o:�o�o�D���D���D���D�����
���
�ě��49X�D����o��o��o��o��o��C���C���t����㼛�㼛�㼛�㼛�㼛�㼬1��9X��j��j��j�ě����ͼ��ͼ���������/��/��`B��`B��`B��h�����o�+�C��C��C��t�����w��w�#�
�#�
�#�
�''0 Ž<j�<j�@��H�9�H�9�H�9�]/�q���y�#�y�#�}�}�}󶽁%��%��7L
#/6<HPTJ</#
  #)0<>A<80+#        ,/:<HMUY`aXUHB</+',,./9;HJT^TSH;/)%(....��������������������������������������������������fmnz���������zmkdcff�������



�������#/<=<6/# ������������������������������������T\hkuuvuh\QPTTTTTTTT���)6BLB:)���������������������������������������Tamz���������maTMLMT�������������������������������������������)5BNRC=5)����� )5BDB>;54)$OP[`hjrsh[YSQOOOOOOO)BN[fkjd[NB:+)�������������������%)-5BNSQRNB5,)%%%%%%^j~��������������sg^
#''0<HTWH</&
���������������������������������������������+, ���������
#+)#
������������������������
���������	

#,/1/#
						BBMOR[fdc[OBBBBBBBBBGIIRUZ]^WURIGGGGGGGG0;DHIHHA;93000000000%%#��������������������g��������������tdabg"/5;?A?<;0/"lmwz������������zmjl2=Ndt��������gNB8,,2)67A6)(��������������������JQOWgw���������tg[LJ��� "+37BMB;)����~������������~~~~~~TUabdcaVUQTTTTTTTTTTBKN[gt}���tgNB@;9<B��
#/<HLLI>/#
����HMUanz�����ha\UQOIHH#2LU^bdleZXOJ<0##nt�����wtrohnnnnnnnn����������������������������������������46BOXURQOB6521444444������������������������������������������������������������egjt���������tgebbce����
#/,$#
�������������������������������22.'!
������#)/5875//'#!#<UbhkjghbU<.'������������������������	


����������
#7IUbfeffbUM<0#
�����������������������


�����������������������������~��������������{{~~*/<U[aejnpsnaUH<4/+*��������!)36<A@86)$������������������������

���������<CGHKLH@<:4356<<<<<<ŹŮŮųŻ������������	������������Ź�/�+�"��!�"�/�9�;�?�;�2�/�/�/�/�/�/�/�/�A�<�4�4�2�4�;�A�M�Z�f�h�i�f�c�^�Z�M�A�A������������������
����
�������������� �������������$�=�?�D�@�=�0�$��ƎƊƎƚƣƳ��������������������ƳƧƚƎ�M�C�S�Y�d������¼׼�߼׼��������f�Y�MÓÈÇ�}�ÇÇÇÓàìôù��ûùìàÓÓ��������ùööù�������������������������H�>�<�D�H�J�U�\�Z�U�H�H�H�H�H�H�H�H�H�H�����������ƾʾξԾξʾ������������������U�N�H�N�T�Z�g�s���������������������g�U�����������������������������������������"���/�6�;�D�H�S�T�\�m����|�m�a�H�;�"�f�d�a�d�o������������¾���������s�i�f���s�~�����������������������ƧƓƅ�|�_�\�V�hƎƧ������������������Ƨ������ŻŶŶŹ��������3�6�K�M�@�6�������s�o�l�s�{����������������������������i�Z�N�E�E�G�F�C�N�Z�s�����������������i�����������������������������������������L�L�Q�Y�b�e�r�~�����~�r�e�Y�Y�L�L�L�L�L����������������������������������������@�4�'�����4�@�M�Y�f�r�w�~�t�r�Y�M�@��޾׾о˾Ѿ׾��������������������ʾ��ʾӾ�����!�(�(����	����¿µ­¦²¿������������������������¿�ʼ¼������ļʼҼּݼ�ּʼʼʼʼʼʼʼʿm�`�O�F�C�?�G�T�T�`�m�y�������������y�m�S�Y�t��������.�9�8�,����ּ������f�S����������������������������úòðôù����������(��������ҾM�C�A�9�4�0�4�A�M�R�W�T�M�M�M�M�M�M�M�M�t�s�o�t�y�;�:�.�)�"��"�.�;�G�G�;�;�;�;�;�;�;�;�;�A�5�4�(�!�(�4�A�M�O�M�F�A�A�A�A�A�A�A�A������(�)�5�7�5�(���������ŭūŤūŭŹ������������žŹŭŭŭŭŭŭ�0�*�#� �#�0�<�>�I�U�b�c�c�b�U�I�<�<�0�0�l�Z�T�Q�T�a�m�z���������������������z�l�����������������������������������������H�E�;�9�5�/�*�-�;�T�a�l�a�Y�W�V�^�\�T�H�/�#��
�����#�/�U�d�o�o�u�k�i�a�H�<�/����������������������������������������������������������������)�������������������-�5�7�E�@�A�5�)���g�>�,�5�A�N�g�s����������������������²«¦¦¨²¿¿��¿²²²²²²�N�I�L�N�Z�g�i�g�f�Z�N�N�N�N�N�N�N�N�N�N���	����*�8�C�L�O�]�`�[�Q�O�C�6�*��������������|�x�z�����������������������;�/� ���%�/�;�H�Q�T�a�j�m�u�m�a�T�H�;�����g�U�N�G�N�Z�s�������������������������������������������������������������һлλû»ûлܻ���������ܻлллллпT�G�-�"��"�(�;�G�T�m�t���������y�m�`�T���
�
���'�4�9�=�5�4�'��������W�O�C�@�W�`�h�tĐĚĦĳķĽ��ĿĚ�t�h�W��׺ʺ׺�����-�F�S�_�j�_�S�F�-����}�q�r�u�~���������úɺ˺ʺź����������}Ň�}�{�s�p�{ŇŔŠŬŭŹ����ſŹŭŠŔŇ�Ŀ��������������ĿѿѿտܿӿѿɿĿĿĿĺr�p�e�Y�L�L�L�R�Y�e�r�z�~�����~�r�r�r�rED�D�D�EEEE*E7EPE\EuE�E�E�EuEiEPE7EE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��û��������������л����(� ����ܻÿĿ��¿Ŀʿѿݿ�����������ݿѿĿĿĿĽĽ��������Ľннݽ߽�ݽڽнĽĽĽĽĽĽ��������~�|���������Ľ׽ϽĽ������������V�O�I�F�I�O�V�Y�b�o�{Ǉǆ�{�o�b�V�V�V�V�ֺԺкֺ����������������ֺ������������������$�-�4�2�.�$��������������������ʾ׾ھ���׾ʾ�������FFFFFFFF$F.F1F=FJFQFTFPFIF=F5F$F�G�:�(��!�.�:�S�`�l�y�����������y�`�S�G�������������Ľнݽ�����ݽѽнĽ�������������������������������������������������ĿķķĻĿ��������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{D{D{D{D{ E = / 3 9 a Z 7  0 : - D H , 8 P R W D : j O 6 0 U S = n \ G C R Y J B Q 6 i . L k * p n X 9 \ C  X - � H \ E . D n 4 9 N G c e E = 7 n Y K = H . e S + ' q    R  I  *      �  �  :  �  A  g     �  �    x  /  �  r  
  �  �    �  �  �  �  k  C  �  �    �  �  �  7  y  �  �  �  �  �  �  M  x  j  �  \    )  �  �  �  u  �  E  �     	  �  �  G  �  V  7  <    b  Z  �    G  4  �  ,    �  �  Ѽ���;��
�ě���o�e`B�o��;d���㼋C��o�ě��H�9�t����C����
�8Q�aG��+�8Q������/�H�9�'ě��H�9�,1�ě���h��1��`B��C���`B�+���������o�C��T���49X�\)��1�\)���49X��9X�o�C��Y��}�Y��',1�D����O߽ixս��P��hs�}�aG��]/�L�ͽ��u��%��C��Y���t���o��+��\)������-���
��7L������
���TB�B%�]B�A�p�B)#BvYBsA��B�|B*�B4��B�cB2�BiB!�SBAB $�BÓB(BB wB\ABOB"��B��BUB)B IB+&B-Z0B+PB^{BP�BzYB�B'2_A�r�B�}BS�B
ȾA��2B ��B	GkB]B)�B
mRB��BAB=BйB[|B}.B&8�B	�B-B߫BO?B6kBLB��B
�B��B گBbUB��B&g�B�?B#åB&�B)�B#�YB�B�Bs�B�BH�B}ZB��B� B��B&6_B�A�nyB;�B��B�,A��(B�<B��B4��B�8B28@B��B!�B>�A�}cB�1B?�BK�BɷBBB7&B"�lB�pB1)B�|B .PB*��B-z�B�B? By�B�;B��B':kA���B��B>wB
�kA�z�B3�B	C�B@�B)~;B
��B��B5�B��BEBB>(B��B&|�B	��B�B�fB>�B��BB�B��B	��B�'B �FB��B��B%b�B�B#�8B'�BB7B#��B�VB��B?0B��BHaB�$B��B�iA�lXA�^2A=:�A�}bB	�KB�@�L�A�`0A��A���AOceA�P|AK>�A�(�AH8AG�B9A�QA�U$A�A���?As@Ԩ*AT�AXM�A�T@�k�Aj��@�C�A.FUA�_aA;A���Aa�A9��A���A��*A��A�f�A��?A���A���A��SA�
EA�#�A��A�mA�r�B ]RA�!qA�YA���A��z@�ڢAf�`@��A݅�@iY�@	|A���Ax��?��C���C�5�@�&�A|�GA(]�A!�Bn�@P��B		fAN�C���Ae�A(�A��]A�_oC�όA�a�A�{�A<��A�~�B	��B� @���A�{eA���A�AO'�A��hAJ�.A�JAH#OAG.B}�A��A�JKA��QA��?�ہAs�@�ѭAU�AY�nA�Q@�?4Al�A�A/�A��A;�yA���Ab��A:��A�w�A�|JA�rA�^A���A�[eA�c�AЇA��FA�rA���A�WiA�~�B I\A�}�A���A�	�A�h�@��zAg@�z�A܄h@p&
@��A��Ax��?�~�C��C�(�@���A}WA(z2A!�RBK�@S��B�AL��C�эAg�A'#�A���A�w.C���   %               
   l               -            	      (               !                     B      /                     	         	   9            ;                                 #                   K                        
               '                           1                     %         +   %      !                  -            =                              #         %         %   +            !      !               #   !               '      '         !                                                   !                     #         )                                       9                              #         !         %   %                  !               #                  '      '                                       O���N!�O�N��O,�O��O�;�NȢ�NxmN&�&NC�O���N	SO��O[�"NvK�PWBOq
�O��Oh�WN�EwNUβO�TO{�N�=~Oa�O�ND�BOF��Pd�'N��O	!	Nn��NJVZN]��N	CN]<�Nh�N�(O��OI�>O62�O�e,M��NbK�O�2O��N-�|M�V�Op��O�tXOj0"O�r�NcXN�C�O�hN�ûOơYO^ͳO�lOSȹN�~xN��OŁ�N�j�O���NߓLN^��N�A�N��N�(�O
u�O DhOskOzׂN�W�N;�xN�9�Nrql  �  �  �  �  �  �    6  N  �  ?  D  �  �  L  �  6  �  �  �    �  �  :  9  �  �  �  �  �    b  �  �  �    a  �  �  �  D  �  �     �  �  �  ]  �  �  �  D  �  �  �  �  <  j  D  �  c  �    J  �  '  �  )    �  �  ,  �  V  �  ?  �  �  �<#�
<o;�`B;��
��o:�o�,1�ě��ě��D���D���e`B�ě���`B�T���D����C�����C����ͼ�o��t����ͼ�t���������ͼ��㼣�
��1��1�#�
��j��j��j�ě����ͼ�������������/��/�'�`B��`B��h�t����o�\)�t��C��C��t�����w��w�'D���49X�''0 ŽD���<j�@��H�9�H�9�ixս]/�q���y�#�y�#�}�}�}󶽟�w��%��7L
#/6<HPTJ</#
  #)0<>A<80+#        ,/:<HMUY`aXUHB</+',,)/;HHSOH;/*&))))))))�������������������������������������������		��������imsz������}zqmgeiiii������������������#/<=<6/# ����������������������������������������Z\hhrmh\WSZZZZZZZZZZ���)4B@9)��������������������������������������Tamz����������maTNNT������������������������������������������)5<;751)��� )5BDB>;54)$Q[_hiqoh[ZTRQQQQQQQQ*5:BMN[`fe][NDB55,**�������������������%)-5BNSQRNB5,)%%%%%%��������������������
"#,/<>HNNH</#
������������������������������������������������**&������
#+)#
������������������������
���������	

#,/1/#
						BBMOR[fdc[OBBBBBBBBBGIIRUZ]^WURIGGGGGGGG0;DHIHHA;93000000000" 
��������������������g��������������tdabg"/5;?A?<;0/"lmwz������������zmjl4:BN[t�������tg[B954)67A6)(��������������������JQOWgw���������tg[LJ�(06:<0)�������~������������~~~~~~TUabdcaVUQTTTTTTTTTT>BN[goty||tlg[NB<;>���
#/<EE<6/#
����HMUanz�����ha\UQOIHH#2LU^bdleZXOJ<0##nt�����wtrohnnnnnnnn����������������������������������������46BOXURQOB6521444444������������������������������������������������������������egjt���������tgebbce����
#/,$#
�������������������������������/0-'
�������#)/5875//'#!#<UbhkjghbU<.'������������������������	


����������04<FIUYbbbacbXUG<800�����������������������


�����������������������������~��������������{{~~*/<U[aejnpsnaUH<4/+*��������!)36<A@86)$������������������������

���������<CGHKLH@<:4356<<<<<<ŹŮŮųŻ������������	������������Ź�/�+�"��!�"�/�9�;�?�;�2�/�/�/�/�/�/�/�/�A�<�4�4�2�4�;�A�M�Z�f�h�i�f�c�^�Z�M�A�A�������������
����
������������������������
����$�0�5�=�@�=�=�3�0�$�ƎƊƎƚƣƳ��������������������ƳƧƚƎ��r�j�h�h�n�v�����������Ǽͼ̼¼������ÓÌÇÀÅÇÓàêìùûùøìàÓÓÓÓùøøù����������������ùùùùùùùù�H�>�<�D�H�J�U�\�Z�U�H�H�H�H�H�H�H�H�H�H�����������ƾʾξԾξʾ������������������W�U�Y�^�g�s�����������������������s�g�W�����������������������������������������"���/�7�<�E�H�T�a�m�}�������z�m�H�;�"�l�f�h�r�����������������������������l���s�~�����������������������ƩƕƆ�~�u�b�c�uƁƎƧ����������������Ʃ������������������������"�$��������������s�q�n�s�~�������������������������s�g�Z�Q�N�K�K�N�Z�g�s�x���������������s�����������������������������������������Y�S�Y�c�e�r�~����~�r�e�Y�Y�Y�Y�Y�Y�Y�Y�����������������������������������������@�4�'�����4�@�M�Y�f�r�w�~�t�r�Y�M�@��޾׾о˾Ѿ׾�����������������۾پ־ھ�����	������	� �����¿´²²¿����������������������������¿�ʼ¼������ļʼҼּݼ�ּʼʼʼʼʼʼʼʿy�m�`�P�G�E�A�G�L�T�]�`�m�y�����������y������f�\�v�������!�.�7�7�*���ּ�������������������������������üþ�������������������������������žM�C�A�9�4�0�4�A�M�R�W�T�M�M�M�M�M�M�M�M�t�s�o�t�y�;�:�.�)�"��"�.�;�G�G�;�;�;�;�;�;�;�;�;�A�5�4�(�!�(�4�A�M�O�M�F�A�A�A�A�A�A�A�A������(�)�5�7�5�(���������ŹŲŭŦŭŭŹ��������ŻŹŹŹŹŹŹŹŹ�0�*�#� �#�0�<�>�I�U�b�c�c�b�U�I�<�<�0�0�l�Z�T�Q�T�a�m�z���������������������z�l�����������������������������������������H�E�;�9�5�/�*�-�;�T�a�l�a�Y�W�V�^�\�T�H�<�/�'�����#�/�:�H�U�e�k�m�j�`�U�H�<����������������������������������������������������������������)�������������������-�5�7�E�@�A�5�)�G�3�5�A�N�g�s�����������������������s�G²«¦¦¨²¿¿��¿²²²²²²�N�I�L�N�Z�g�i�g�f�Z�N�N�N�N�N�N�N�N�N�N�*������"�*�6�C�O�Y�\�]�[�Y�O�C�6�*�����������������������������������������;�/� ���%�/�;�H�Q�T�a�j�m�u�m�a�T�H�;�����g�U�N�G�N�Z�s�������������������������������������������������������������һлλû»ûлܻ���������ܻлллллпT�G�-�"��"�(�;�G�T�m�t���������y�m�`�T���
�
���'�4�9�=�5�4�'��������Y�O�D�@�X�a�hĎĚĦĲĶļľĳĦĚ�t�h�Y������������!�-�F�T�X�N�F�:�-���������~�t�z�~�������������º�������������Ň�}�{�s�p�{ŇŔŠŬŭŹ����ſŹŭŠŔŇ�Ŀ��������������ĿѿѿտܿӿѿɿĿĿĿĺr�p�e�Y�L�L�L�R�Y�e�r�z�~�����~�r�r�r�rED�D�D�EEEE*E7EPE\EuE�E�EuEiEPE7EEE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��û��������������л����(� ����ܻÿĿ��¿Ŀʿѿݿ�����������ݿѿĿĿĿĽĽ��������Ľннݽ߽�ݽڽнĽĽĽĽĽĽ��������������������������ĽĽ����������V�O�I�F�I�O�V�Y�b�o�{Ǉǆ�{�o�b�V�V�V�V�ֺԺкֺ����������������ֺ������������������$�-�4�2�.�$��������������������ʾ׾ھ���׾ʾ�������FFFFFFFF$F.F1F=FJFQFTFPFIF=F5F$F�G�:�(��!�.�:�S�`�l�y�����������y�`�S�G�������������Ľнݽ�����ݽѽнĽ�������������������������������������������������ĿķķĻĿ��������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{D{D{D{D{ E = / + 3 a ? : & 0 : ! A K $ 8 J 0 X / : R 9 6 0 % X = n X G # R Y J B Q 3 i . L k 1 p n X 0 \ C  J - � H \ E . ? g . 9 N G c e E = 7 7 Y K = H . e S ( ' q    R  I  *  �  V  �  �  �    A  g  5    �  �  x  �  �  E  �  �  �  A  �  �  �  W  k    �  �  ,  �  �  �  7  y  n  �  �  �  �  �  M  x  j  K  \    �  P  �  �  u  �  E  �  �    C  �  G  �  #  7  <    b  (  �    G  4  �  ,    N  �  �  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  �  �  �  �  �  �  �  c  =    �  �  Q     �  J  �  k    �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  r  j  c  [  T  L  �  �  �  �  �  �  �  s  Y  =     �  �  �  �  c  3  �  �  G  �  �  �  �  �  v  f  T  B  0      �  �  �    M    �  
  �  �  �  �  �  �  �  �  �  �  �  �  p  A  	  �  �  6  �    �  �  �  �  �  �  �  �  �  b  ?    �  �  �  �  s  Q  2    	@  	�  	�  
0  
�  
�        
�  
�  
#  	�  	3  �  �  I  8    8     )  0  4  5  0  &      �  �  �  �  [    �  S  �  p   �    +  8  D  N  I  B  9  -  !      �  �  �  �  �  d  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ?  <  :  7  5  3  0  )            �   �   �   �   �   �   �   �  �  �    >  D  6    �  �  �  �  c  3  �  �  |  (  �  K  |  �  �  �  �  �  �  }  u  m  f  ^  V  M  E  =  1  %       �  �  �  �  �  �  �  {  i  �  r  h  [  N  8    �  q    �  X  �  G  K  D  5    �  �  �  �  {  ^  C  '    �  �  B  �  h  �  �  �  �  �  �  �  �  �  �  �  p  R  4    �  �  �  �  m  /  3  '  /  !    �  �  �  �  �  �  [  .  �  �  |  8  �  O  <  q  �  �  �  �  �  �  �  �  �  a  #  �  �  3  �  B  p  W  �  �  �  �  �  �  �  �  �  {  l  X  ?    �  �    8  �  �  �  �  �  �  �  �  �  �  �  �  �  c  *  �  �  V  �  �  �  Z      �  �  �  �  �  �  �  �  �  y  h  X  <            x  �  �  �  �  x  `  C  &    �  �  �  �  e  F  <  7  0  )  (  R  l  �  �  �  �  }  c  F  "  �  �  �  >  �  H  ]  E    :  -      �  �  �  �  �  w  p  f  U  =    �  �  9  �  :  9  +        �  �  �  �  �  �  �  l  Q  6       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  /  �  �  f  �  �    <  _  �  �  �  �  m  V  6      �  �  �    a  �  �  �  �  �  �  �  �  �  �  z  q  i  `  P  <  (    �  �  �  �  �  �  �  �  �  �  �  x  g  R  <  $    �  �  �  �  U   �   �  �  �  �  �  �  �  ~  U  ,  
  �  �  �  6  �  l  �  4  �  �      �  �  �  �  �  �  �  �  �  �  g  F    �  �  �  @   �  K  �  �    &  @  R  `  `  [  E  &  �  �  h  �      
  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  [  M  @  3  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  �  �  �  w  l  b  W  M  B  7  -  "         �   �   �   �   �   �   �    �  �  �  �  �  �  �  |  j  V  B  -      �  �  �  �  �  a  P  ?  -      �  �  �  �  �    e  G  *     �   �   �   }  �  �  �  �  �  �  �  �  |  h  S  >  %    �  �  �  �  v  V  �  �  �  x  W  6    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  d  K  /    �  �  �  �  �  �  �  �  �  �    U  !  �  D  B  ?  6  '      �  �  �  �  ]  ;    �  �  �  ;  �  �  �  �  �  �  �  �  z  j  S  :    �  �  �  �  {  y  d  1   �    z  �  �  �  �  �  �  o  1  �  �  &  �  P  �  &  �  �  ]     �  �  �  �  �  �  w  ^  A  #  
      �  �  �  �  {  Y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  Y  1  �  �  �  �  )  J  �  �  �  x  m  k  k  w  {  i  B    �  h  �  h  �  �  �  ]  R  H  >  4  *            �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  }  z  y  y  y  x  x  w  w  w  v  �  �  �  �  �  �  �  �  �  �  p  I    �  �  �  j  J  �  �  �  �  �  �  �  �  r  Q  *    �  �  �  �  �  k    p  �   �  D  D  A  9  +      �  �  �  �  q  B    �  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  o  [  A  '       �   �   �   �  �  �  �  �  �  �  �  �  |  n  a  S  E  5  "    �  �  t  3  �  �  �  �  �  ^  8    �  �  �  �  �  W  #  �  �  �  F    �  �  �  n  V  C  -       �  �  �  �  �  4  �  �    b   �  <  5  -  !       �  �  �  Z  &  �  �  6  �  Y  �  �  ,  �  [  g  Q  +  �  �  �  e  ,  �  �  u  -  �  t  �  L  �     �  �    2  <  B  D  >  6  +      �  �  �  ;  �  �    '  *  �  �  �  �  �  �  �  �  �  c  :  	  �  �  ^    �    "  �  c  Z  P  D  8  *      �  �  �  y  K  :  ,    �  �  p   �  �  �  �  �  �  �  �  �  �  �  i  F     �  �  �  ?    �  �      �  �  �  �  �  �  �  u  ^  G  4       %  1  0  ,  '  ;  C  (  
�  
�  
�  
�  
�  
�  =    
�  
�  
@  	�  	1  �  �  K    �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  G  ?  L  x  '      �  �  �  �  �  }  ]  5      �  �      *  b  �  �  �  �  �  w  ^  D  &    �  �  �  c  +  �  �  4  �  �  /  )  "          �  �  �  �  �  �  �  �  ~  k  W  C  /    �  �  �  �  �      �  �  �  n  5  �  �  x  2  �  �  M    �  }  l  T  :      �  �  �  �  c  =    �  �  G    �  �  �  �  �  �  y  p  b  S  C  6  )    
  �  �  �  �  �  �  T  ,      �  �  �  �  �  �  c  D  "  �  �  �  }  R  (     �  �  �  �  �  �  �  �  �  w  l  ^  M  9  #      �  �  �  �  V  <    �  �  �  �  \  /    �  �  K  �  �  K  �  n  �  �  �  z  e  M  2    �  �  �  �  �  �  p  I    �  �    �  +  ?  7  /  '        �  �  �  �  �  �  �  �  s  �  �  �  �  E  ?  T  ?         �  �  �    4  
T  
/  	x  �  �  �  �  �  �  �  �  ~  m  X  @  #  �  �  �  J  �  �  5  �  M  �  O  �  �  �  t  =    �  �  i  +  �  �  �  F    �  �  Z  !  �  �